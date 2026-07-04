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
import numpy as np
import torch
from sklearn.cluster import DBSCAN

# maybe add:
# from sklearn.metrics import (
#     f1_score,
#     balanced_accuracy_score,
#     matthews_corrcoef,
#     cohen_kappa_score
# )

from utils.logger import print_log


class SegMetricsTracker:
    """Tracks segmentation metrics across epochs."""

    def __init__(
        self,
        num_seg_classes: int,
        seg_classes_map: dict = None,
        category_names: list = None,
        track_object_metrics: bool = False,
        cluster_eps: float = 0.3,       # DBSCAN Radius (30cm)
        cluster_min_samples: int = 20  # at least points for object
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


    def new_accumulator(self):
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
            # acc for 3D obj-metrics
            'obj_tp_count': 0,
            'obj_true_count': 0,
            'obj_pred_count': 0,
            'obj_total_ious': []
        }
        return acc

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_object_wise_3d(self, xyz: np.ndarray, pred_labels: np.ndarray, target_labels: np.ndarray, manhole_label: int = 1):
        # mask for manhole-points
        pred_mask = (pred_labels == manhole_label)
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

            clustered_pred = DBSCAN(
                eps=self.cluster_eps,
                min_samples=self.cluster_min_samples,
                algorithm='kd_tree',
                n_jobs=1
            ).fit(xyz_pred_cluster)

            # -1 label is noise
            pred_obj_ids = clustered_pred.labels_
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
                sampled_target_indices = np.random.choice(target_indices, MAX_DBSCAN_POINTS, replace=False)
                xyz_target_cluster = xyz[sampled_target_indices]
            else:
                sampled_target_indices = target_indices
                xyz_target_cluster = xyz[sampled_target_indices]

            clustered_target = DBSCAN(
                eps=self.cluster_eps, 
                min_samples=self.cluster_min_samples,
                algorithm='kd_tree',
                n_jobs=1
            ).fit(xyz_target_cluster)

            target_obj_ids = clustered_target.labels_
            valid_target = target_obj_ids != -1
            num_true_objects = len(np.unique(target_obj_ids[valid_target]))
            
            target_cluster_values = np.where(valid_target, target_obj_ids + 1, 0)
            full_target_obj_ids[sampled_target_indices] = target_cluster_values

        # 3. Matching
        tp_objects = 0
        all_obj_ious = []
        used_pred_ids = set()

        for obj_idx in range(1, num_true_objects + 1):
            true_obj_mask = (full_target_obj_ids == obj_idx)
            pred_ids_in_target = full_pred_obj_ids[true_obj_mask]
            pred_ids_in_target = pred_ids_in_target[pred_ids_in_target > 0]

            matched_pred_mask = None
            obj_iou = 0.0

            if len(pred_ids_in_target) > 0:
                counts = np.bincount(pred_ids_in_target)
                unique_preds_sorted = np.argsort(counts)[::-1]
                
                best_pred_id = None
                for p_id in unique_preds_sorted:
                    if p_id > 0 and counts[p_id] > 0 and p_id not in used_pred_ids:
                        best_pred_id = p_id
                        break
                
                if best_pred_id is not None:
                    used_pred_ids.add(best_pred_id)
                    matched_pred_mask = (full_pred_obj_ids == best_pred_id)
                    
                    union = np.sum(true_obj_mask | matched_pred_mask)
                    inter = np.sum(true_obj_mask & matched_pred_mask)
                    obj_iou = inter / float(union) if union > 0 else 0.0

            all_obj_ious.append(obj_iou)
            if obj_iou >= 0.3:  # iou_threshold
                tp_objects += 1

        num_fp_objects = max(0, num_pred_objects - len(used_pred_ids))
        all_obj_ious.extend([0.0] * num_fp_objects)

        return tp_objects, num_true_objects, num_pred_objects, all_obj_ious

    # def evaluate_object_wise_3d(self, xyz: np.ndarray, pred_labels: np.ndarray, target_labels: np.ndarray, manhole_label: int = 1):
    #     # mask for manhole-points
    #     pred_mask = (pred_labels == manhole_label)
    #     target_mask = (target_labels == manhole_label)

    #     # 1. Get pred objects via clustering
    #     xyz_pred = xyz[pred_mask]
    #     if len(xyz_pred) > 0:
    #         clustered_pred = DBSCAN(
    #             eps=self.cluster_eps,
    #             min_samples=self.cluster_min_samples
    #         ).fit(xyz_pred)
    #         # -1 label is noise
    #         pred_obj_ids = clustered_pred.labels_
    #         valid_pred = pred_obj_ids != -1
    #         num_pred_objects = len(np.unique(pred_obj_ids[valid_pred]))

    #         full_pred_obj_ids = np.zeros_like(pred_labels)
    #         full_pred_obj_ids[pred_mask] = np.where(valid_pred, pred_obj_ids+1, 0)
    #     else:
    #         full_pred_obj_ids = np.zeros_like(pred_labels)
    #         num_pred_objects = 0

    #     # 2. Get gt objects via clustering
    #     xyz_target = xyz[target_mask]
    #     if len(xyz_target) > 0:
    #         clustered_target = DBSCAN(
    #             eps=self.cluster_eps, 
    #             min_samples=self.cluster_min_samples
    #         ).fit(xyz_target)
    #         target_obj_ids = clustered_target.labels_
    #         valid_target = target_obj_ids != -1
    #         num_true_objects = len(np.unique(target_obj_ids[valid_target]))
            
    #         full_target_obj_ids = np.zeros_like(target_labels)
    #         full_target_obj_ids[target_mask] = np.where(valid_target, target_obj_ids + 1, 0)
    #     else:
    #         full_target_obj_ids = np.zeros_like(target_labels)
    #         num_true_objects = 0

    #     # 3. Matching
    #     tp_objects = 0
    #     all_obj_ious = []
    #     used_pred_ids = set()

    #     for obj_idx in range(1, num_true_objects + 1):
    #         true_obj_mask = (full_target_obj_ids == obj_idx)
    #         pred_ids_in_target = full_pred_obj_ids[true_obj_mask]
    #         pred_ids_in_target = pred_ids_in_target[pred_ids_in_target > 0]

    #         matched_pred_mask = None
    #         obj_iou = 0.0

    #         if len(pred_ids_in_target) > 0:
    #             # Hole die IDs sortiert nach Häufigkeit (absteigend)
    #             counts = np.bincount(pred_ids_in_target)
    #             unique_preds_sorted = np.argsort(counts)[::-1]
                
    #             # Finde die häufigste Prädiktions-ID, die noch NICHT verwendet wurde
    #             best_pred_id = None
    #             for p_id in unique_preds_sorted:
    #                 if p_id > 0 and counts[p_id] > 0 and p_id not in used_pred_ids:
    #                     best_pred_id = p_id
    #                     break
                
    #             if best_pred_id is not None:
    #                 used_pred_ids.add(best_pred_id)
    #                 matched_pred_mask = (full_pred_obj_ids == best_pred_id)
                    
    #                 union = np.sum(true_obj_mask | matched_pred_mask)
    #                 inter = np.sum(true_obj_mask & matched_pred_mask)
    #                 obj_iou = inter / float(union) if union > 0 else 0.0

    #         all_obj_ious.append(obj_iou)
    #         if obj_iou >= 0.3:  # iou_threshold
    #             tp_objects += 1

    #         # if len(pred_ids_in_target) > 0:
    #         #     best_pred_id = np.argmax(np.bincount(pred_ids_in_target))

    #         #     if best_pred_id in used_pred_ids:
    #         #         all_obj_ious.append(0.0)
    #         #         continue

    #         #     used_pred_ids.add(best_pred_id)

    #         #     matched_pred_mask = (full_pred_obj_ids == best_pred_id)
    #         #     union = np.sum(true_obj_mask | matched_pred_mask)
    #         #     inter = np.sum(true_obj_mask & matched_pred_mask)
    #         #     obj_iou = inter / float(union) if union > 0 else 0.0
    #         #     all_obj_ious.append(obj_iou)

    #         #     if obj_iou >= 0.3:  # iou_threshold
    #         #         tp_objects += 1
    #         # else:
    #         #     all_obj_ious.append(0.0)

    #     # add FPs (predictions which could not match up on any GT object)
    #     num_fp_objects = max(0, num_pred_objects - len(used_pred_ids))
    #     all_obj_ious.extend([0.0] * num_fp_objects)

    #     return tp_objects, num_true_objects, num_pred_objects, all_obj_ious


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
                tp_o, true_o, pred_o, ious_o = self.evaluate_object_wise_3d(xyz_np[i], segp, segl, MANHOLE_LABEL)
                accumulator['obj_tp_count'] += tp_o
                accumulator['obj_true_count'] += true_o
                accumulator['obj_pred_count'] += pred_o
                accumulator['obj_total_ious'].extend(ious_o)

            # Part-Seg Klassiker (ShapeNet)
            if self.seg_classes_map is not None:
                cat_name = self.seg_label_to_cat.get(int(segl[0]), None)
                if cat_name.strip() and cat_name in self.seg_classes_map:
                    parts = self.seg_classes_map[cat_name]
                    part_ious = [ (np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l))) if np.sum((segl == l) | (segp == l)) > 0 else 1.0) for l in parts]
                    accumulator['shape_ious'].setdefault(cat_name, []).append(float(np.mean(part_ious)))

            # if self.seg_classes_map is not None:
            #     cat_name = self.seg_label_to_cat.get(int(segl[0]), None)
            #     if cat_name is None:
            #         continue
            #     parts = self.seg_classes_map[cat_name]
            #     part_ious = []
            #     for l in parts:
            #         inter = np.sum((segl == l) & (segp == l))
            #         union = np.sum((segl == l) | (segp == l))
            #         if union == 0:
            #             part_ious.append(1.0)
            #         else:
            #             part_ious.append(inter / float(union))
            #     accumulator['shape_ious'].setdefault(cat_name, []).append(
            #         float(np.mean(part_ious))
            #     )
            # else:
            #     # Semantic seg: accumulate global TP/FP/FN per class.
            #     # This matches the PointNet/PointNet++ S3DIS evaluation protocol
            #     # and avoids the per-block averaging bias.
            #     for l in range(self.num_seg_classes):
            #         pred_l = segp == l
            #         true_l = segl == l
            #         accumulator['tp'][l] += int(np.sum(pred_l & true_l))
            #         accumulator['fp'][l] += int(np.sum(pred_l & ~true_l))
            #         accumulator['fn'][l] += int(np.sum(~pred_l & true_l))

    # def new_accumulator(self):
    #     acc = {'total_correct': 0, 
    #            'total_seen': 0, 
    #            'shape_ious': {},
    #            'manhole_correct': 0,
    #            'manhole_seen': 0}
    #     if self.seg_classes_map is None:
    #         # Semantic seg: global TP/FP/FN per class (correct S3DIS protocol).
    #         acc['tp'] = np.zeros(self.num_seg_classes, dtype=np.int64)
    #         acc['fp'] = np.zeros(self.num_seg_classes, dtype=np.int64)
    #         acc['fn'] = np.zeros(self.num_seg_classes, dtype=np.int64)
    #     return acc

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
            if accumulator['total_seen'] > 0 else 0.0
        )

        manhole_accuracy = (
            accumulator['manhole_correct'] / float(accumulator['manhole_seen'])
            if accumulator['manhole_seen'] > 0 else 0.0
        )

        # pixel-based precision / recall for manhole class (index 1)
        tp_pts = accumulator['tp'][1]
        fp_pts = accumulator['fp'][1]
        fn_pts = accumulator['fn'][1]
        precision = tp_pts / (tp_pts + fp_pts) if (tp_pts + fp_pts) > 0 else 0.0
        recall = tp_pts / (tp_pts + fn_pts) if (tp_pts + fn_pts) > 0 else 0.0

        # calc obj-metrics if tracked
        obj_precision = 0.0
        obj_recall = 0.0
        obj_f1 = 0.0
        obj_mean_iou = 0.0
        
        if self.track_object_metrics:
            tp_o = accumulator['obj_tp_count']
            true_o = accumulator['obj_true_count']
            pred_o = accumulator['obj_pred_count']
            
            obj_precision = tp_o / pred_o if pred_o > 0 else 0.0
            obj_recall = tp_o / true_o if true_o > 0 else 0.0
            obj_f1 = (2 * obj_precision * obj_recall) / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0.0
            
            # Sichere Abfrage für den Mittelwert der Objekt-IoUs
            if len(accumulator['obj_total_ious']) > 0:
                obj_mean_iou = float(np.mean(accumulator['obj_total_ious']))

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
            'accuracy': accuracy * 100.0,
            'manhole_accuracy': manhole_accuracy * 100.0,
            'instance_miou': instance_miou * 100.0,
            'class_miou': class_miou * 100.0,
            'per_category_iou': {k: v * 100.0 for k, v in per_cat.items()},
            'precision': precision,
            'recall': recall,
            'obj_precision': obj_precision,
            'obj_recall': obj_recall,
            'obj_f1': obj_f1,
            'obj_mean_iou': obj_mean_iou * 100.0
        }

        # manhole_precision = accumulator['tp'][1] / (accumulator['tp'][1] + accumulator['fp'][1])
        # manhole_recall = accumulator['tp'][1] / (accumulator['tp'][1] + accumulator['fn'][1])

        # if self.seg_classes_map is None:
        #     # ── Semantic seg: global TP/FP/FN ──────────────────────────────
        #     tp = accumulator['tp']
        #     fp = accumulator['fp']
        #     fn = accumulator['fn']
        #     denom = tp + fp + fn
        #     iou_per_class = np.where(denom > 0, tp / denom.astype(np.float64), np.nan)
        #     # Only count classes that appear in the evaluation set
        #     valid = ~np.isnan(iou_per_class)
        #     miou = float(np.mean(iou_per_class[valid])) if valid.any() else 0.0
        #     per_cat = {
        #         str(c): float(iou_per_class[c])
        #         for c in range(self.num_seg_classes)
        #         if not np.isnan(iou_per_class[c])
        #     }
        #     return {
        #         'accuracy': accuracy * 100.0,
        #         'manhole_accuracy': manhole_accuracy * 100.0,
        #         'instance_miou': miou * 100.0,
        #         'class_miou': miou * 100.0,
        #         'per_category_iou': {k: v * 100.0 for k, v in per_cat.items()},
        #         'precision': manhole_precision,
        #         'recall': manhole_recall
        #     }
        # else:
        #     # ── Part seg: per-shape IoU average ────────────────────────────
        #     all_ious = []
        #     per_cat = {}
        #     for cat, ious in accumulator['shape_ious'].items():
        #         all_ious.extend(ious)
        #         per_cat[cat] = float(np.mean(ious)) if ious else 0.0
        #     instance_miou = float(np.mean(all_ious)) if all_ious else 0.0
        #     class_miou = float(np.mean(list(per_cat.values()))) if per_cat else 0.0
        #     return {
        #         'accuracy': accuracy * 100.0,
        #         'manhole_accuracy': manhole_accuracy * 100.0,
        #         'instance_miou': instance_miou * 100.0,
        #         'class_miou': class_miou * 100.0,
        #         'per_category_iou': {k: v * 100.0 for k, v in per_cat.items()},
        #         'precision': manhole_precision,
        #         'recall': manhole_recall
        #     }

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
