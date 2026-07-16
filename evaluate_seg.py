"""
Call me like this:
python evaluate_seg.py \
    --config cfgs/segmentation/CurveNet/WHUUrban3D/curvenet_whuurban3D.yaml \
    --ckpts experiments/manhole_seg_curvenet_whu_first_try/ckpt-best-seg.pth
"""
from pathlib import Path

import os
import time
import warnings

warnings.filterwarnings('ignore', message='.*adaptive_max_pool2d_backward_cuda.*')
warnings.filterwarnings('ignore', message='.*TORCH_CUDA_ARCH_LIST.*')
warnings.filterwarnings('ignore', message='.*weights_only=False.*')

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np

import torch
import torch.nn as nn

from tools import builder
from tools.runner_seg import _validate
from tools.metrics_tracker_seg import SegMetricsTracker

from utils import parser, misc
from utils.logger import get_root_logger, print_log
from utils.config import get_config, log_args_to_file, log_config_to_file


from pathlib import Path
import sys

# --- IMPORTANT: make project root importable when calling:
# python tools/evaluate_seg.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# helper
def get_data_for_plot(obj_results, 
                      fix_metric_name="iou_threshold", fix_value=0.5, 
                      variable_metric_name="confident_threshold"):
    """
    Filtert die Daten so, dass eine Metrik fixiert bleibt (z.B. iou_threshold = 0.5)
    und die andere Metrik über die X-Achse läuft.
    """
    # filter after threshols
    if fix_metric_name == "confident_threshold":
        key_idx = 0
    elif fix_metric_name == "iou_threshold":
        key_idx = 1
    else:
        raise ValueError(f"fix_metric_name must be in ['confident_threshold', 'iou_threshold'] but got: {fix_metric_name}")

    filtered = {key: value for key, value in obj_results.items() if key[key_idx] == fix_value}

    # convert to list
    filtered_list = []
    for key, value in filtered_dict.items():
        item = value.copy() 
        item.update({
            'confident_threshold': key[0], 
            'iou_threshold': key[1]
        })
        filtered_list.append(item)
        # sort for clean line
        
    filtered_list.sort(key=lambda x: x[variable_metric_name])

    x_thresholds = [r[variable_metric_name] for r in filtered_list]
    
    # directly calc here, we only have the micro aggregations here
    AP = []
    AR = []
    AIOU = []

    for r in filtered_list:
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        iou = r["iou_sum"] / max(tp, 1) if tp > 0 else 0.0
        
        AP.append(precision)
        AR.append(recall)
        AIOU.append(iou)

    return np.array(x_thresholds), np.array(AP), np.array(AR), np.array(AIOU)



def plot_and_save(iou_thresholds, AR, AP, AIOU, title, xlabel, save_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

    # plot values
    ax.plot(iou_thresholds, AR, label='Avg Recall (AR)', marker='o', linewidth=2)
    ax.plot(iou_thresholds, AP, label='Avg Precision (AP)', marker='s', linewidth=2)
    # ax.plot(iou_thresholds, AF1, label='Avg F1-Score (AF1)', marker='^', linewidth=2)
    if AIOU is not None:
        ax.plot(iou_thresholds, AIOU, label='Avg Object IoU', marker='d', linewidth=2)

    # set titles and labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)

    # set range for these metrics
    ax.set_ylim(0, 1.05)

    # add grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', fontsize=11)

    # save and close fig
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)



def plot_mean_average(object_results, root_save_path, save_name):
    # 1. Confidence Curve (Fixed IoU Threshold on 0.5, variable Confident Threshold)
    save_path1 = os.path.join(root_save_path, f"confidence_curve_{save_name}.png")
    
    conf_thresholds, AP, AR, AIOU = get_data_for_plot(
        object_results, 
        fix_metric_name="iou_threshold", 
        fix_value=0.5, 
        variable_metric_name="confident_threshold"
    )

    plot_and_save(
        iou_thresholds=conf_thresholds, 
        AR=AR, 
        AP=AP, 
        AIOU=AIOU,
        title="Metrics vs. Confidence Threshold (Fix IoU @ 0.5)",
        xlabel='Confidence Threshold',
        save_path=save_path1
    )

    # 2. IoU Matching Sensitivity Curve (Fixed Confident Threshold on 0.5, variable IoU Threshold)
    save_path2 = os.path.join(root_save_path, f"matching_sensitivity_curve_{save_name}.png")
    
    iou_thresholds, AP, AR, AIOU = get_data_for_plot(
        object_results, 
        fix_metric_name="confident_threshold", 
        fix_value=0.5, 
        variable_metric_name="iou_threshold"
    )

    plot_and_save(
        iou_thresholds=iou_thresholds, 
        AR=AR, 
        AP=AP, 
        AIOU=AIOU,
        title="Metrics vs. IoU Matching Threshold (Fix Conf @ 0.5)",
        xlabel='IoU Matching Threshold',
        save_path=save_path2
    )

# main eval
def evaluate(args, config):

    # Dataset
    _, test_loader = builder.dataset_builder(
        args,
        config.dataset.test
    )

    test_dataset = test_loader.dataset

    print(f"Test size: {len(test_dataset)}")
    # raise ValueError("Debug Stop")

    num_seg_classes = getattr(
        test_dataset,
        "num_seg_classes",
        config.model.seg_classes
    )

    num_obj_classes = getattr(
        test_dataset,
        "num_obj_classes",
        config.model.num_obj_classes
    )

    category_names = getattr(
        test_dataset,
        "category_names",
        None
    )

    use_cls_label = config.model.get(
        "use_cls_label",
        False
    )

    if use_cls_label:
        seg_classes_map = getattr(
            test_dataset,
            "seg_classes",
            None
        )
    else:
        seg_classes_map = None

    # Model
    model = builder.model_builder(config.model)

    checkpoint = torch.load(
        args.ckpts,
        map_location="cpu"
    )

    parts = Path(args.ckpts).parts
    exp_name = parts[-2]

    state_dict = checkpoint["base_model"]

    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)

    model = nn.DataParallel(model).cuda()
    model.eval()

    # Metrics
    metrics_tracker = SegMetricsTracker(
        num_seg_classes=num_seg_classes,
        seg_classes_map=seg_classes_map,
        category_names=category_names,
        track_object_metrics=True, 
        cluster_eps=0.2,
        cluster_min_samples=50,
        confident_thresholds=np.arange(0.50, 0.95+0.05, 0.05),
        iou_thresholds=np.arange(0.00, 0.95+0.05, 0.05)
    )

    metrics = _validate(
        model,
        test_loader,
        metrics_tracker,
        num_obj_classes=num_obj_classes,
        use_cls_label=use_cls_label,
    )

    output_lines = []
    output_lines.append("="*40)
    output_lines.append(f"QUANTITATIVE TEST RESULTS ({exp_name})")
    output_lines.append("="*40)
    
    for metric_name, value in metrics.items():
        # remove prefix 'test_' or 'eval_'
        clean_name = metric_name.replace("test_", "").replace("eval_", "")
        if clean_name == 'obj_entry':  # isinstance(value, dict):
            line = f"Object Results on different Confidence & IoU:"
            output_lines.append(line)
            for (conf_threshold, iou_threshold), value_dict in value.items():
                line = f"  - Confidence >= {conf_threshold} & IoU >= {iou_threshold}:"
                output_lines.append(line)
                
                tp_objects = value_dict['tp']
                fp_objects = value_dict['fp']
                fn_objects = value_dict['fn']
                iou_sum = value_dict['iou_sum']
                
                obj_iou = iou_sum / tp_objects if tp_objects > 0 else 1.0
                obj_precision = tp_objects / max(tp_objects+fp_objects, 1)
                obj_recall = tp_objects / max(tp_objects+fn_objects, 1)
                obj_f1 = (2 * obj_precision * obj_recall) / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0.0
             
                line = f"      > 'IoU': {obj_iou}"
                output_lines.append(line)
                line = f"      > 'Precision': {obj_precision}"
                output_lines.append(line)
                line = f"      > 'Recall': {obj_recall}"
                output_lines.append(line)
                line = f"      > 'F1': {obj_f1}"
                output_lines.append(line)
        else:
            line = f"{clean_name:<30}: {value:.4f}" if isinstance(value, float) else f"{clean_name:<30}: {value}"
            output_lines.append(line)
        
    output_lines.append("="*40)
    
    # print
    text_content = "\n".join(output_lines)
    print(text_content)
    
    eval_path_root = "./tmp_eval"
    os.makedirs(eval_path_root, exist_ok=True)
    txt_path = os.path.join(eval_path_root, f"test_metrics_{exp_name}.txt")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_content)
        
    print(f"Metrics successfully saved to: {txt_path}")

    plot_mean_average(
        object_results=metrics["eval_obj_entry"],   # or test_ ?
        root_save_path=eval_path_root, 
        save_name=exp_name
    )




def main():
    args = parser.get_args()

    # basic runtime flags
    args.use_gpu = torch.cuda.is_available()

    # reproducibility (optional, same style as training main)
    if getattr(args, "seed", None) is not None:
        misc.set_random_seed(
            args.seed + getattr(args, "local_rank", 0),
            deterministic=getattr(args, "deterministic", False)
        )

    # make sure there is always some experiment path
    if not hasattr(args, "experiment_path") or args.experiment_path is None:
        args.experiment_path = "./experiments_eval"

    os.makedirs(args.experiment_path, exist_ok=True)

    # utput dir fallback
    # if not hasattr(args, "output_dir") or args.output_dir is None:
    #     args.output_dir = os.path.join(args.experiment_path, "eval_results")

    # os.makedirs(args.output_dir, exist_ok=True)

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}_eval.log')
    logger = get_root_logger(
        log_file=log_file,
        name=getattr(args, "log_name", "eval")
    )

    # config
    config = get_config(args, logger=logger)

    # optional args/config logging
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)

    # batch size fallback for test set if config has no explicit bs
    if hasattr(config, "dataset") and config.dataset.get("test"):
        if getattr(config.dataset.test.others, "bs", None) is None:
            config.dataset.test.others.bs = 16

    evaluate(args, config)



if __name__ == "__main__":
    main()







