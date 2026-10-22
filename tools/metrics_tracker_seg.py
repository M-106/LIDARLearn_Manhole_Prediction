"""
Segmentation Metrics Tracker.

Tracks per-point accuracy and mean Intersection-over-Union (mIoU) for
point cloud segmentation. Computes two mIoU variants standard for
ShapeNet Parts evaluation:
    - Instance-average mIoU: per-shape IoU averaged over all shapes.
    - Class-average mIoU: per-shape IoU averaged within each category,
                          then averaged across categories.

The best model is selected by instance-average mIoU (standard convention).
"""

import csv
import os
import shutil
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from utils.logger import print_log

# from helper import save_dir_creation



# helper
def calculate_polygon_iou(xyz_1, xyz_2):
    # projection on xy plane
    hull_1 = ConvexHull(xyz_1[:, :2])
    hull_2 = ConvexHull(xyz_2[:, :2])

    poly_1 = Polygon(xyz_1[hull_1.vertices, :2])
    poly_2 = Polygon(xyz_2[hull_2.vertices, :2])

    if not poly_1.is_valid:
        poly_1 = poly_1.buffer(0)
        poly_2 = poly_2.buffer(0)

    intersection = poly_1.intersection(poly_2).area
    union = poly_1.union(poly_2).area

    return intersection / union if union > 0 else 0.0


# main Metric Object/Logic
class SegMetricsTracker:
    """Tracks segmentation metrics across epochs."""

    def __init__(
        self,
        num_seg_classes: int,
        seg_classes_map: dict = None,
        category_names: list = None,
        track_object_metrics: bool = True,
        cluster_eps: float = 0.3,       # DBSCAN Radius (30cm)
        cluster_min_samples: int = 20,  # at least points for object
        confident_thresholds=np.arange(0.50, 0.95+0.05, 0.05),  # FIXME -> right?
        iou_thresholds=np.arange(0.00, 0.95+0.05, 0.05)
    ):
        """
        Args:
            num_seg_classes: total number of part/semantic classes (50 for ShapeNet Parts).
            seg_classes_map: dict mapping category name → list of part label ids,
                e.g. {'Airplane': [0,1,2,3], 'Chair': [12,13,14,15], ...}
                Required for ShapeNet Parts-style mIoU. If None, only semantic
                mIoU (single category, all classes) is computed.
            category_names: optional list of category names for logging.
        """
        self.num_seg_classes = num_seg_classes
        self.seg_classes_map = seg_classes_map
        self.category_names = category_names or (
            list(seg_classes_map.keys()) if seg_classes_map else ['all']
        )

        # object metrics settings
        self.track_object_metrics = track_object_metrics
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.confident_thresholds = confident_thresholds
        self.iou_thresholds = iou_thresholds

        if isinstance(self.confident_thresholds, np.ndarray):
            self.confident_thresholds = self.confident_thresholds.round(2)

        if isinstance(self.iou_thresholds, np.ndarray):
            self.iou_thresholds = self.iou_thresholds.round(2)

        # Build reverse map: part_label → category name
        self.seg_label_to_cat = {}
        if seg_classes_map:
            for cat, labels in seg_classes_map.items():
                for lab in labels:
                    self.seg_label_to_cat[lab] = cat

        # Best metrics
        self.best_instance_miou = 0.0
        self.best_class_miou = 0.0
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.per_category_iou_at_best = {}

        # History
        self.history = {
            'epoch': [], 'train_loss': [], 'train_acc': [], 'train_manhole_acc': [],
            'train_precision': [], 'train_recall': [],
            'val_loss': [], 'val_acc': [], 'val_manhole_acc': [], 
            'val_class_miou': [], 'val_instance_miou': [],
            'val_precision': [], 'val_recall': [],
            'val_obj_precision': [], 'val_obj_recall': [], 'val_obj_f1': [], 'val_obj_mean_iou': []
        }


    def new_accumulator(self, confident_thresholds=np.arange(0.50, 0.95+0.05, 0.05), iou_thresholds=np.arange(0.0, 0.95+0.05, 0.05)):
        if isinstance(confident_thresholds, np.ndarray):
            confident_thresholds = confident_thresholds.round(2)

        if isinstance(iou_thresholds, np.ndarray):
            iou_thresholds = iou_thresholds.round(2)
        
        acc = {
            'total_correct': 0, 
            'total_seen': 0, 
            'shape_ious': {},
            'manhole_correct': 0,
            'manhole_seen': 0,
            # for global pixel-metrics (Precision/Recall via TP/FP/FN)
            'tp': np.zeros(self.num_seg_classes, dtype=np.int64),
            'fp': np.zeros(self.num_seg_classes, dtype=np.int64),
            'fn': np.zeros(self.num_seg_classes, dtype=np.int64),
            # acc/detection for 3D obj-metrics
            # 'obj_tp_count': 0,
            # 'obj_true_count': 0,
            # 'obj_pred_count': 0,
            # 'obj_total_ious': []
            'obj_metrics': {(cp, it): {'tp': 0, 'fp': 0, 'fn': 0, 'pred': 0, 'true': 0, 'iou_sum': 0}
                            for cp in confident_thresholds for it in iou_thresholds},
            'obj_mA_recall': 0,
            'obj_mA_precision': 0,
            'obj_mA_iou': 0,
            'avg_true_objects_per_img': 0,
            'avg_pred_objects_per_img': 0
        }
        return acc

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_object_wise_3d(self, xyz: np.ndarray, pred_labels: np.ndarray, target_labels: np.ndarray, manhole_label: int = 1, confident_threshold=0.5, iou_threshold=0.5):
        # print("DEBUGGING 3D CHECKPOINT 1", flush=True)
        # mask for manhole-points
        # pred_mask = (pred_labels == manhole_label)
        pred_mask = (pred_labels > confident_threshold)
        target_mask = (target_labels == manhole_label)

        MAX_CLUSTER_POINTS = int(20000)  # int(20000 * 0.25)

        # 1. Get pred objects via clustering
        pred_indices = np.where(pred_mask)[0]
        num_pred_pts = len(pred_indices)
        
        full_pred_obj_ids = np.zeros_like(pred_labels)
        num_pred_objects = 0
        
        if num_pred_pts > 0:

            # reduce points else clustering will explode the memory
            if num_pred_pts > MAX_CLUSTER_POINTS:
                sampled_indices = np.random.choice(pred_indices, MAX_CLUSTER_POINTS, replace=False)
                
                xyz_pred_cluster = xyz[sampled_indices]
            else:
                sampled_indices = pred_indices
                xyz_pred_cluster = xyz[sampled_indices]

            # print("DEBUGGING 3D CHECKPOINT 2", flush=True)
            clustered_pred = DBSCAN(
                eps=self.cluster_eps,
                min_samples=self.cluster_min_samples
            ).fit(xyz_pred_cluster)
            # print("DEBUGGING 3D CHECKPOINT 3", flush=True)

            # -1 label is noise
            pred_obj_ids = clustered_pred.labels_
            MIN_POINTS = 10
            labels, counts = np.unique(pred_obj_ids, return_counts=True)
            # Find all Cluster-IDs (but not -1), which have not enough points
            small_clusters = labels[(counts <= MIN_POINTS) & (labels != -1)]
            
            # Re-label Points in small clusters as -1
            if len(small_clusters) > 0:
                pred_obj_ids[np.isin(pred_obj_ids, small_clusters)] = -1
            
            valid_pred = pred_obj_ids != -1
            num_pred_objects = len(np.unique(pred_obj_ids[valid_pred]))

            cluster_values = np.where(valid_pred, pred_obj_ids+1, 0)
            full_pred_obj_ids[sampled_indices] = cluster_values

        # 2. Get gt objects via clustering
        target_indices = np.where(target_mask)[0]
        num_target_pts = len(target_indices)
        
        full_target_obj_ids = np.zeros_like(target_labels)
        num_true_objects = 0
        
        if num_target_pts > 0:
            if num_target_pts > MAX_CLUSTER_POINTS:
                sampled_target_indices = np.random.choice(target_indices, MAX_CLUSTER_POINTS, replace=False)
                xyz_target_cluster = xyz[sampled_target_indices]
            else:
                sampled_target_indices = target_indices
                xyz_target_cluster = xyz[sampled_target_indices]

            # print("DEBUGGING 3D CHECKPOINT 4", flush=True)
            clustered_target = DBSCAN(
                eps=self.cluster_eps, 
                min_samples=self.cluster_min_samples
            ).fit(xyz_target_cluster)
            # print("DEBUGGING 3D CHECKPOINT 5", flush=True)

            target_obj_ids = clustered_target.labels_
            valid_target = target_obj_ids != -1
            num_true_objects = len(np.unique(target_obj_ids[valid_target]))
            
            target_cluster_values = np.where(valid_target, target_obj_ids + 1, 0)
            full_target_obj_ids[sampled_target_indices] = target_cluster_values

        # Check number of objects -> catch 0 object cases
        if num_true_objects == 0 and num_pred_objects == 0:
            return 0, 0, 0, 0, 0, 0

        if num_true_objects == 0 and num_pred_objects > 0:
            # Only FPs
            return 0, 0, num_pred_objects, 0, 0, 0

        if num_true_objects > 0 and num_pred_objects == 0:
            # Only FNs
            return 0, num_true_objects, 0, 0, 0, 0

        # print("DEBUGGING 3D CHECKPOINT 6", flush=True)
        # plot example - FIXME comment me out after checking
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Live Metric Debug (GT Objs: {num_true_objects}, Pred Objs: {num_pred_objects})", fontsize=14)

        # col 1: original image masks comparison
        # DEBUG pred_mask: (200000,)
        # DEBUG target_mask: (200000,)
        # DEBUG overlay_raw: Shape (200000, 3), for removing z axis
        # Shape (full_pred_obj_ids): (200000,)
        # Shape (full_target_obj_ids): (200000,)
        # Shape (target_labels): (200000,)
        # Shape (pred_labels): (200000,)
        # Shape (target_mask): (200000,)
        # Shape (pred_mask): (200000,)
        # Shape (xyz_target_cluster): (11595, 3)
        # Shape xyz: (200000, 3)
        # try:
        #     print(f"DEBUG pred_mask: {pred_mask.shape}")
        #     print(f"DEBUG target_mask: {target_mask.shape}")
        #     print(f"DEBUG overlay_raw: Shape {(*pred_mask.shape, 3)}, for removing z axis")
        #     print(f"Shape (full_pred_obj_ids): {full_pred_obj_ids.shape}")
        #     print(f"Shape (full_target_obj_ids): {full_target_obj_ids.shape}")
        #     print(f"Shape (target_labels): {target_labels.shape}")
        #     print(f"Shape (pred_labels): {pred_labels.shape}")
        #     print(f"Shape (target_mask): {target_mask.shape}")
        #     print(f"Shape (pred_mask): {pred_mask.shape}")
        #     print(f"Shape (xyz_target_cluster): {xyz_target_cluster.shape}")
        #     print(f"Shape xyz: {xyz.shape}")
        # except Exception as e:
        #     print(f"Got error: {e}")

        overlay_raw = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        overlay_raw[target_mask > 0.5] = [255, 0, 0]  # GT = red
        overlay_raw[pred_mask > 0.5] = [0, 255, 0]  # preds = green
        overlay_raw[(target_mask > 0.5) & (pred_mask > 0.5)] = [255, 255, 0]  # intersection = yellow

        x = xyz[:, 0]
        y = xyz[:, 1]

        plot_mask = (target_mask > 0.5) | (pred_mask > 0.5)
        if not np.any(plot_mask):
            plot_mask = np.ones_like(target_mask, dtype=bool)

        color_rgb = overlay_raw[plot_mask] / 255.0
        axes[0].scatter(x[plot_mask], y[plot_mask], c=color_rgb, s=2)
        axes[0].set_title("Original Masks (red=GT, green=pred)")
        axes[0].set_aspect('equal')

        # col 2: with closing operation + poylgon (come a bit later)
        axes[1].scatter(x=x, y=y, c=full_target_obj_ids, cmap='tab20', s=1)
        axes[1].set_title("GT after Clustering")
        axes[1].set_aspect('equal')
        axes[2].scatter(x=x, y=y, c=full_pred_obj_ids, cmap='tab20', s=1)
        axes[2].set_title("Pred after Clustering")
        axes[2].set_aspect('equal')

        file_root_path = "./tmp_eval/tmp_viz"
        # save_dir_creation(file_root_path)
        os.makedirs(file_root_path, exist_ok=True)
        shutil.rmtree(file_root_path)
        # save_dir_creation(file_root_path)
        os.makedirs(file_root_path, exist_ok=True)
        plt.savefig(os.path.join(file_root_path, f"tmp_eval_clustering_check_{np.random.randint(0, 1000)}.png"))
        plt.close(fig)

        # print("DEBUGGING 3D CHECKPOINT 7", flush=True)


        # 3. Costmatrix for matching
        cost_matrix = np.ones((num_true_objects, num_pred_objects), dtype=np.float32)
        
        for t_idx in range(1, num_true_objects+1):
            true_mask = (full_target_obj_ids == t_idx)
            xyz_target = xyz[true_mask]

            for p_idx in range(1, num_pred_objects+1):
                pred_mask = (full_pred_obj_ids == p_idx)
                xyz_pred = xyz[pred_mask]

                iou = calculate_polygon_iou(xyz_target, xyz_pred)
                cost_matrix[t_idx - 1, p_idx - 1] = 1.0 - iou

        # 4. Gloal Matching with cost matrix
        row_indeces, col_indeces = linear_sum_assignment(cost_matrix)

        # TP/FP/FN counting
        tp_objects = 0
        iou_sum = 0.0
        all_obj_pred_ious = []
        used_pred_ids = set()

        for t_i, p_i in zip(row_indeces, col_indeces):
            iou = 1.0 - cost_matrix[t_i, p_i]
            if iou >= iou_threshold:
                iou_sum += iou
                used_pred_ids.add(p_i+1)
                tp_objects += 1

        fp_objects = num_pred_objects - len(used_pred_ids)
        fn_objects = num_true_objects - len(row_indeces)

        # print("DEBUGGING 3D CHECKPOINT 8", flush=True)
        
        return tp_objects, fp_objects, fn_objects, num_true_objects, num_pred_objects, iou_sum



    def evaluate_batch(self, pred_logits: np.ndarray, target: np.ndarray,
                       shape_categories: np.ndarray, accumulator: dict, xyz=None):
        """Accumulate per-shape IoUs for one batch.

        Args:
            pred_logits: [B, N, num_seg_classes] numpy array (or B,C,N transposed)
            target:      [B, N] int labels
            shape_categories: [B] int indicating each shape's object category
                              (only used to restrict valid parts for part seg).
            accumulator: dict with keys 'total_correct', 'total_seen',
                         'shape_ious' (list of per-shape ious by category).
        """
        B, N = target.shape
        # Accept [B, N, C] or [B, C, N]
        if pred_logits.shape[1] == N and pred_logits.shape[2] == self.num_seg_classes:
            # [B, N, C] already
            logits_bnc = pred_logits
        else:
            logits_bnc = pred_logits.transpose(0, 2, 1)  # [B, C, N] -> [B, N, C]

        cur_pred = np.zeros((B, N), dtype=np.int32)
        for i in range(B):
            if self.seg_classes_map is not None:
                # Part seg: restrict argmax to the valid parts of this shape's category.
                cat_idx = shape_categories[i]
                # Resolve category name: either directly from category_names[cat_idx],
                # or via target's first label through seg_label_to_cat.
                if 0 <= cat_idx < len(self.category_names):
                    cat_name = self.category_names[cat_idx]
                else:
                    cat_name = self.seg_label_to_cat.get(int(target[i, 0]), None)
                if cat_name is not None and cat_name in self.seg_classes_map:
                    valid = np.array(self.seg_classes_map[cat_name])
                    local_pred = np.argmax(logits_bnc[i][:, valid], axis=1)
                    cur_pred[i, :] = valid[local_pred]   # correct for non-contiguous labels
                    continue
            # Fallback: full argmax (semantic seg or unknown category)
            cur_pred[i, :] = np.argmax(logits_bnc[i], axis=1)

        accumulator['total_correct'] += int(np.sum(cur_pred == target))
        accumulator['total_seen'] += int(B * N)

        # Per-shape IoU
        MANHOLE_LABEL = 1

        xyz_np = None
        if self.track_object_metrics and xyz is not None:
            if isinstance(xyz, torch.Tensor):
                xyz_np = xyz.detach().cpu().numpy()
            else:
                xyz_np = np.array(xyz)
            
            if xyz_np.shape[1] == 3 and xyz_np.shape[2] == N:
                xyz_np = xyz_np.transpose(0, 2, 1)
           
            xyz_np = xyz_np[:, :, :3]

        for i in range(B):
            segp = cur_pred[i]
            segl = target[i]

            mask = (segl == MANHOLE_LABEL)
            accumulator['manhole_seen'] += int(np.sum(mask))
            accumulator['manhole_correct'] += int(np.sum((segp == segl) & mask))

            # global semantic TP/FP/FN counting for stabile metrics
            for l in range(self.num_seg_classes):
                pred_l = (segp == l)
                true_l = (segl == l)
                accumulator['tp'][l] += int(np.sum(pred_l & true_l))
                accumulator['fp'][l] += int(np.sum(pred_l & ~true_l))
                accumulator['fn'][l] += int(np.sum(~pred_l & true_l))

            # 3D object metrics
            if self.track_object_metrics and xyz_np is not None:
                for confident_threshold in self.confident_thresholds:
                    for iou_threshold in self.iou_thresholds:
                        # tp_objects, fp_objects, fn_objects, num_true_objects, num_pred_objects, iou_sum = self.evaluate_object_wise_3d(xyz_np[i], segp, segl, MANHOLE_LABEL, confident_threshold, iou_threshold)
                        
                        tp_objects = 1
                        fp_objects = 1
                        fn_objects = 1
                        num_true_objects = 1 
                        num_pred_objects = 1
                        iou_sum = 1

                        # accumulator['obj_tp_count'] += tp_o
                        # accumulator['obj_true_count'] += true_o
                        # accumulator['obj_pred_count'] += pred_o
                        # accumulator['obj_total_ious'].extend(ious_o)
                        key = (confident_threshold, iou_threshold)
                        metrics = accumulator['obj_metrics'][key]
                        metrics['tp'] += tp_objects
                        metrics['fp'] += fp_objects
                        metrics['fn'] += fn_objects
                        metrics['true'] += num_true_objects
                        metrics['pred'] += num_pred_objects
                        metrics['iou_sum'] += iou_sum
                         

            # Part-Seg Klassiker (ShapeNet)
            if self.seg_classes_map is not None:
                cat_name = self.seg_label_to_cat.get(int(segl[0]), None)
                if cat_name.strip() and cat_name in self.seg_classes_map:
                    parts = self.seg_classes_map[cat_name]
                    part_ious = [ (np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l))) if np.sum((segl == l) | (segp == l)) > 0 else 1.0) for l in parts]
                    accumulator['shape_ious'].setdefault(cat_name, []).append(float(np.mean(part_ious)))

    

    def finalize(self, accumulator: dict):
        """Compute mIoU from an accumulator.

        Part segmentation (seg_classes_map is not None):
            - instance_miou: mean over all shapes of per-shape mean-IoU.
            - class_miou: mean over categories of per-category mean-IoU.
            Both follow the standard ShapeNet Parts convention.

        Semantic segmentation (seg_classes_map is None):
            - Uses GLOBAL TP/FP/FN accumulated across ALL points (PointNet++
              S3DIS protocol).  Per-block averaging inflates mIoU by giving
              equal weight to small and large blocks, which can shift results
              by 1-4 pp versus published baselines.
            - instance_miou and class_miou are both the global mIoU
              (they are identical in the single-category semantic case).
        """
        accuracy = (
            accumulator['total_correct'] / float(accumulator['total_seen'])
            if accumulator['total_seen'] > 0 else 1.0
        )

        manhole_accuracy = (
            accumulator['manhole_correct'] / float(accumulator['manhole_seen'])
            if accumulator['manhole_seen'] > 0 else 1.0
        )

        # pixel-based precision / recall for manhole class (index 1)
        tp_pts = accumulator['tp'][1]
        fp_pts = accumulator['fp'][1]
        fn_pts = accumulator['fn'][1]
        precision = tp_pts / (tp_pts + fp_pts) if (tp_pts + fp_pts) > 0 else 0.0
        recall = tp_pts / (tp_pts + fn_pts) if (tp_pts + fn_pts) > 0 else 0.0

        # calc obj-metrics if tracked

        tp_objects = -999
        fp_objects = -999
        fn_objects = -999
        obj_precision = -999
        obj_recall = -999
        obj_f1 = -999
        obj_iou = -999
        num_true_objects = -999
        num_pred_objects = -999

        if self.track_object_metrics:
            tp_objects = 0
            fp_objects = 0
            fn_objects = 0
            num_true_objects = 0
            num_pred_objects = 0
            total_iou_sum = 0.0

            # for confidence_threshold, iou_threshold in accumulator['obj_metrics'].keys():
            #     if confidence_threshold >= 0.5 and iou_threshold == 0.5:
            obj_entry = accumulator['obj_metrics'][(0.5, 0.5)]
            tp_objects += obj_entry['tp']
            fp_objects += obj_entry['fp']
            fn_objects += obj_entry['fn']
            num_true_objects += obj_entry['true']
            num_pred_objects += obj_entry['pred']
            total_iou_sum += obj_entry['iou_sum']
    
            obj_iou = total_iou_sum / max(tp_objects, 1)
            obj_precision = tp_objects / max(tp_objects+fp_objects, 1)
            obj_recall = tp_objects / max(tp_objects+fn_objects, 1)
            obj_f1 = (2 * obj_precision * obj_recall) / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0.0
             
            # obj_precision = tp_o / pred_o if pred_o > 0 else 0.0
            # obj_recall = tp_o / true_o if true_o > 0 else 0.0
            # obj_f1 = (2 * obj_precision * obj_recall) / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0.0
            
            # Sichere Abfrage für den Mittelwert der Objekt-IoUs
            # if len(accumulator['obj_total_ious']) > 0:
            #     obj_mean_iou = float(np.mean(accumulator['obj_total_ious']))

        # semantic or instance mIoU
        if self.seg_classes_map is None:
            denom = accumulator['tp'] + accumulator['fp'] + accumulator['fn']
            iou_per_class = np.where(denom > 0, accumulator['tp'] / denom.astype(np.float64), np.nan)
            valid = ~np.isnan(iou_per_class)
            miou = float(np.mean(iou_per_class[valid])) if valid.any() else 0.0
            per_cat = {str(c): float(iou_per_class[c]) for c in range(self.num_seg_classes) if not np.isnan(iou_per_class[c])}
            instance_miou = class_miou = miou
        else:
            all_ious = []
            per_cat = {}
            for cat, ious in accumulator['shape_ious'].items():
                all_ious.extend(ious)
                per_cat[cat] = float(np.mean(ious)) if ious else 0.0
            instance_miou = float(np.mean(all_ious)) if all_ious else 0.0
            class_miou = float(np.mean(list(per_cat.values()))) if per_cat else 0.0

        return {
            'accuracy': accuracy,
            'manhole_accuracy': manhole_accuracy,
            'instance_miou': instance_miou,
            'class_miou': class_miou,
            'per_category_iou': {k: v for k, v in per_cat.items()},
            'precision': precision,
            'recall': recall,
            'obj_entry': accumulator['obj_metrics'],
            'obj_precision': obj_precision,
            'obj_recall': obj_recall,
            'obj_f1': obj_f1,
            'obj_iou': obj_iou,
            'num_true_objects': num_true_objects,
            'num_pred_objects': num_pred_objects,
            'tp_objects': tp_objects,
            'fp_objects': fp_objects,
            'fn_objects': fn_objects
        }            




    # ------------------------------------------------------------------
    # Update best
    # ------------------------------------------------------------------

    def update(self, epoch: int, metrics: dict, selection_metric: str = 'instance_miou') -> bool:
        """Update best-metrics tracking. Selection criterion is configurable."""
        current = metrics.get(selection_metric, metrics['instance_miou'])
        best_so_far = getattr(self, f'_best_{selection_metric}', 0.0)
        is_best = current > best_so_far
        if is_best:
            setattr(self, f'_best_{selection_metric}', current)
        if is_best:
            self.best_instance_miou = metrics['instance_miou']
            self.best_class_miou = metrics['class_miou']
            self.best_accuracy = metrics['accuracy']
            self.best_epoch = epoch
            self.per_category_iou_at_best = dict(metrics['per_category_iou'])
        return is_best

    def update_history(self,
                       epoch,
                       train_loss=0.0,
                       train_acc=0.0,
                       train_manhole_acc=0.0,
                       train_precision=0.0,
                       train_recall=0.0,
                       val_loss=0.0,
                       val_acc=0.0,
                       val_manhole_acc=0.0,
                       val_class_miou=0.0,
                       val_instance_miou=0.0,
                       val_precision=0.0,
                       val_recall=0.0,
                       val_obj_precision=0.0,
                       val_obj_recall=0.0, 
                       val_obj_f1=0.0, 
                       val_obj_mean_iou=0.0):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['train_manhole_acc'].append(train_manhole_acc)
        self.history['train_precision'].append(train_precision)
        self.history['train_recall'].append(train_recall)

        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_manhole_acc'].append(val_manhole_acc)
        self.history['val_precision'].append(val_precision)
        self.history['val_recall'].append(val_recall)

        self.history['val_class_miou'].append(val_class_miou)
        self.history['val_instance_miou'].append(val_instance_miou)

        self.history['val_obj_precision'].append(val_obj_precision)
        self.history['val_obj_recall'].append(val_obj_recall)
        self.history['val_obj_f1'].append(val_obj_f1)
        self.history['val_obj_mean_iou'].append(val_obj_mean_iou)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def print_best(self, logger=None):
        print_log(
            f"[Best Seg Model] Epoch {self.best_epoch}: "
            f"Acc={self.best_accuracy:.4f}%, "
            f"Class mIoU={self.best_class_miou:.4f}%, "
            f"Instance mIoU={self.best_instance_miou:.4f}%",
            logger=logger,
        )
        if self.per_category_iou_at_best:
            print_log("  Per-category IoU:", logger=logger)
            for cat, iou in sorted(self.per_category_iou_at_best.items()):
                print_log(f"    {cat:<14}: {iou:.4f}%", logger=logger)

    def save_history_csv(self, save_path: str):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.history.keys())
            writer.writerows(zip(*self.history.values()))

    def save_summary_csv(self, save_path: str, model_name: str, dataset: str = ''):
        row = {
            'model_name': model_name,
            'dataset': dataset,
            'best_epoch': self.best_epoch,
            'best_accuracy': f'{self.best_accuracy:.4f}',
            'best_class_miou': f'{self.best_class_miou:.4f}',
            'best_instance_miou': f'{self.best_instance_miou:.4f}',
        }
        for cat, iou in self.per_category_iou_at_best.items():
            row[f'iou_{cat}'] = f'{iou:.4f}'
        file_exists = os.path.exists(save_path)
        with open(save_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
