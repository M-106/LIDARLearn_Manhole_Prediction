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



def evaluate(args, config):

    # Dataset
    _, test_loader = builder.dataset_builder(
        args,
        config.dataset.test
    )

    test_dataset = test_loader.dataset

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
        cluster_min_samples=50
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
        line = f"{clean_name:<30}: {value:.4f}" if isinstance(value, float) else f"{clean_name:<30}: {value}"
        output_lines.append(line)
        
    output_lines.append("="*40)
    
    # print
    text_content = "\n".join(output_lines)
    print(text_content)
    
    os.makedirs("./tmp_eval", exist_ok=True)
    txt_path = os.path.join("./tmp_eval", f"test_metrics_{exp_name}.txt")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_content)
        
    print(f"Metrics successfully saved to: {txt_path}")




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







