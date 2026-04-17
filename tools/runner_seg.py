"""
Segmentation Training Runner.

Two modes, selected by the model's `use_cls_label` flag:
  Part segmentation (use_cls_label=True, e.g. ShapeNet Parts)
  Semantic segmentation (use_cls_label=False, e.g. S3DIS)
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from tools import builder
from utils import dist_utils
from utils.AverageMeter import AverageMeter
from utils.logger import get_logger, print_log
from utils import log_format as fmt

from .metrics_tracker_seg import SegMetricsTracker


def _to_one_hot(y, num_classes):
    if y.dim() == 2 and y.shape[1] == 1:
        y = y.squeeze(1)
    return torch.eye(num_classes, device=y.device)[y.long()]


def _unpack_data(data, use_cls_label):
    if len(data) == 3:
        points, cls_label, target = data
        if not use_cls_label:
            cls_label = None
    elif len(data) == 2:
        points, target = data
        cls_label = None
    else:
        raise ValueError(f"Expected data tuple of length 2 or 3, got {len(data)}")
    return points, cls_label, target


def _validate(model, dataloader, metrics_tracker, num_obj_classes, use_cls_label):
    model.eval()
    acc = metrics_tracker.new_accumulator()
    with torch.no_grad():
        for _, _, data in dataloader:
            points, cls_label, target = _unpack_data(data, use_cls_label)
            points = points.float().cuda()
            target = target.long().cuda()
            points_bcn = points.transpose(1, 2).contiguous()
            if use_cls_label:
                cls_label = cls_label.long().cuda().view(-1)
                cls_onehot = _to_one_hot(cls_label, num_obj_classes)
                logits = model(points_bcn, cls_onehot)
                cls_np = cls_label.detach().cpu().numpy()
            else:
                logits = model(points_bcn, None)
                cls_np = np.zeros(points.shape[0], dtype=np.int32)

            pred_np = logits.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            metrics_tracker.evaluate_batch(pred_np, target_np, cls_np, acc)

    return metrics_tracker.finalize(acc)


def run_net(args, config):
    logger = get_logger(args.log_name)

    # Banner
    fmt.print_banner(args, config, logger=logger)

    # Datasets
    _, train_loader = builder.dataset_builder(args, config.dataset.train)
    _, val_loader = builder.dataset_builder(args, config.dataset.val)

    train_dataset = train_loader.dataset
    num_seg_classes = getattr(train_dataset, 'num_seg_classes', config.model.get('seg_classes', 50))
    num_obj_classes = getattr(train_dataset, 'num_obj_classes', config.model.get('num_obj_classes', 16))
    category_names = getattr(train_dataset, 'category_names', None)

    use_cls_label = config.model.get('use_cls_label', True)
    if use_cls_label:
        seg_classes_map = getattr(train_dataset, 'seg_classes', None)
    else:
        seg_classes_map = None

    # Model
    base_model = builder.model_builder(config.model)

    ckpt_info = None
    if getattr(args, 'ckpts', None) is not None:
        pre_state = {k: v.clone() for k, v in base_model.state_dict().items()}
        if hasattr(base_model, 'load_backbone_ckpt'):
            base_model.load_backbone_ckpt(args.ckpts, strict=False)
        elif hasattr(base_model, 'load_model_from_ckpt'):
            base_model.load_model_from_ckpt(args.ckpts)
        post_state = base_model.state_dict()
        loaded_count = sum(
            1 for k in pre_state
            if k in post_state and not torch.equal(pre_state[k], post_state[k])
        )
        ckpt_info = {'path': args.ckpts, 'loaded': loaded_count, 'total': len(post_state)}
        del pre_state

    if args.use_gpu:
        base_model.cuda()

    if args.distributed:
        if args.sync_bn:
            base_model = nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        base_model = nn.parallel.DistributedDataParallel(
            base_model, device_ids=[args.local_rank % torch.cuda.device_count()]
        )
    else:
        base_model = nn.DataParallel(base_model).cuda()

    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    total_params = sum(p.numel() for p in base_model.module.parameters())
    trainable_params = sum(p.numel() for p in base_model.module.parameters() if p.requires_grad)
    fmt.print_model_info(trainable_params, total_params, ckpt_info, logger=logger)

    # Metrics tracker
    metrics_tracker = SegMetricsTracker(
        num_seg_classes=num_seg_classes,
        seg_classes_map=seg_classes_map,
        category_names=category_names,
    )

    max_epoch = config.max_epoch
    grad_clip = config.get('grad_norm_clip', None)
    selection_metric = config.get('selection_metric', 'instance_miou')
    early_stopping_patience = config.get('early_stopping_patience', 0)
    epochs_without_improvement = 0
    epoch_times = []

    if args.val_freq > max_epoch:
        print_log(
            f"[WARNING] val_freq={args.val_freq} > max_epoch={max_epoch}: "
            "validation will never run and no best checkpoint will be saved.",
            logger=logger,
        )

    fmt.print_training_start(config, selection_metric, early_stopping_patience, logger=logger)

    best_metrics_dict = {}   # persisted to checkpoint so best epoch is recoverable
    base_model.zero_grad()
    for epoch in range(1, max_epoch + 1):
        base_model.train()
        losses = AverageMeter(['loss', 'acc'])
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100, leave=False)
        for _, _, data in pbar:
            points, cls_label, target = _unpack_data(data, use_cls_label)
            points = points.float().cuda()
            target = target.long().cuda()
            points_bcn = points.transpose(1, 2).contiguous()
            if use_cls_label:
                cls_label = cls_label.long().cuda().view(-1)
                cls_onehot = _to_one_hot(cls_label, num_obj_classes)
                logits = base_model(points_bcn, cls_onehot)
            else:
                logits = base_model(points_bcn, None)
            loss, acc = base_model.module.get_loss_acc(logits, target)

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip, norm_type=2)
            optimizer.step()
            base_model.zero_grad()

            losses.update([loss.item(), acc.item()])
            pbar.set_postfix({'loss': f'{losses.avg(0):.4f}', 'acc': f'{losses.avg(1):.1f}%'})

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        elif scheduler is not None:
            scheduler.step(epoch)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        lr = optimizer.param_groups[0]['lr']

        # Validation
        if epoch % args.val_freq == 0:
            metrics = _validate(
                base_model, val_loader, metrics_tracker,
                num_obj_classes=num_obj_classes,
                use_cls_label=use_cls_label,
            )
            is_best = metrics_tracker.update(epoch, metrics, selection_metric=selection_metric)
            metrics_tracker.update_history(
                epoch,
                train_loss=losses.avg(0), train_acc=losses.avg(1),
                val_acc=metrics['accuracy'],
                val_class_miou=metrics['class_miou'],
                val_instance_miou=metrics['instance_miou'],
            )

            tqdm.write(fmt.format_seg_epoch_line(
                epoch, losses.avg(0), losses.avg(1),
                metrics['accuracy'], metrics['instance_miou'],
                metrics['class_miou'], metrics_tracker.best_instance_miou,
                lr, epoch_time, is_best,
            ))

            if is_best:
                epochs_without_improvement = 0
                best_metrics_dict = {
                    'epoch': epoch,
                    'instance_miou': metrics['instance_miou'],
                    'class_miou': metrics['class_miou'],
                    'accuracy': metrics['accuracy'],
                }
                builder.save_checkpoint(
                    base_model, optimizer, epoch,
                    metrics=best_metrics_dict, best_metrics=best_metrics_dict,
                    prefix='ckpt-best-seg', args=args, logger=None,
                )
            else:
                epochs_without_improvement += 1

            builder.save_checkpoint(
                base_model, optimizer, epoch,
                metrics=metrics, best_metrics=best_metrics_dict,
                prefix='ckpt-last-seg', args=args, logger=None,
            )

            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                fmt.print_early_stopping(epoch, selection_metric, early_stopping_patience, logger=logger)
                break

    # Final summary
    fmt.print_seg_summary(metrics_tracker, logger=logger)

    # Save outputs
    model_name = config.model.get('base_model', config.model.NAME)
    history_path = os.path.join(args.experiment_path, f'seg_history_{model_name}.csv')
    metrics_tracker.save_history_csv(history_path)
    summary_path = os.path.join(args.experiment_path, 'seg_summary.csv')
    metrics_tracker.save_summary_csv(
        summary_path, model_name=model_name,
        dataset=config.dataset.train._base_.NAME,
    )

    fmt.print_saved_outputs(args.experiment_path, [
        f'seg_history_{model_name}.csv',
        'seg_summary.csv',
    ], logger=logger)
    fmt.print_training_complete(metrics_tracker.best_instance_miou, logger=logger)

    return {
        'metrics_tracker': metrics_tracker,
        'model_name': model_name,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'avg_epoch_time': sum(epoch_times) / len(epoch_times) if epoch_times else 0.0,
    }
