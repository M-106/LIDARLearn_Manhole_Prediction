"""
Unified Pretraining Runner
Supports PointBERT, ACT, Point-MAE, RECON, PointGPT and other SSL models.
"""

import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from tools import builder
from utils import misc, dist_utils
from utils.logger import get_logger, print_log
from utils.AverageMeter import AverageMeter
from utils import log_format as fmt

from datasets.augmentation import get_train_transforms
from .validation import validate_svm


class Acc_Metric:
    def __init__(self, acc=0.):
        if isinstance(acc, dict):
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        return self.acc > other.acc

    def state_dict(self):
        return {'acc': self.acc}


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    model_name_upper = str(config.model.NAME).upper()
    if 'PCP' in model_name_upper:
        config.dataset.train.others.whole = True

    svm_return_percent = model_name_upper.startswith('ACT')

    # Banner
    fmt.print_banner(args, config, logger=logger)

    # Datasets
    (train_sampler, train_dataloader), (_, test_dataloader) = (
        builder.dataset_builder(args, config.dataset.train),
        builder.dataset_builder(args, config.dataset.val),
    )

    extra_train_dataloader = None
    if config.dataset.get('extra_train'):
        (_, extra_train_dataloader) = builder.dataset_builder(args, config.dataset.extra_train)
    else:
        print_log(
            "[pretrain] No `dataset.extra_train` defined — SVM linear probe "
            "will be skipped for this run. Add an `extra_train` split with a "
            "labelled dataset (e.g. ModelNet40) to monitor encoder quality.",
            logger=logger,
        )

    # Model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.ckpts is not None:
        builder.load_model(base_model, args.ckpts, logger=logger)

    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
    else:
        base_model = nn.DataParallel(base_model).cuda()

    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    total_params = sum(p.numel() for p in base_model.module.parameters())
    trainable_params = sum(p.numel() for p in base_model.module.parameters() if p.requires_grad)
    fmt.print_model_info(trainable_params, total_params, logger=logger)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
        # RNG must be restored LAST — after all weight/optimizer allocations
        builder.resume_rng_state(args, logger=logger)

    # CLI --augmentation (non-'none') overrides the yaml's pretrain_augmentation
    # field. This mirrors the finetune runner's behavior so the same flag works
    # uniformly across all runner entry points.
    cli_aug = getattr(args, 'augmentation', 'none') or 'none'
    if cli_aug != 'none':
        aug_name = cli_aug
    else:
        aug_name = config.get('pretrain_augmentation', 'none')
    train_transforms = get_train_transforms(aug_name)
    fmt.print_augmentation(train_transforms, logger=logger)
    fmt.print_training_start(config, 'svm_acc', 0, logger=logger)

    grad_clip = config.get('grad_norm_clip', None)
    epoch_times = []

    # Training loop
    epoch_pbar = tqdm(
        range(start_epoch, config.max_epoch + 1),
        desc="Pretraining",
        unit="epoch",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
    )

    base_model.zero_grad()
    for epoch in epoch_pbar:
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start = time.time()
        losses = AverageMeter(['loss'])
        num_iter = 0

        batch_pbar = tqdm(
            train_dataloader,
            desc=f"  Epoch {epoch:3d}",
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        )

        for idx, batch in enumerate(batch_pbar):
            num_iter += 1

            if len(batch) == 3:
                taxonomy_ids, model_ids, data = batch
                points = data[0].cuda() if isinstance(data, (list, tuple)) else data.cuda()
            else:
                points = batch.cuda() if not isinstance(batch, (list, tuple)) else batch[0].cuda()

            npoints = config.dataset.train.others.npoints
            if points.size(1) != npoints:
                points = misc.fps(points, npoints)

            points = train_transforms(points)

            loss_output = base_model(points)
            total_loss = sum(loss_output) if isinstance(loss_output, tuple) else loss_output
            total_loss.backward()

            if num_iter == config.step_per_update:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                total_loss = dist_utils.reduce_tensor(total_loss, args)
            losses.update([total_loss.item()])

            if args.distributed:
                torch.cuda.synchronize()

            batch_pbar.set_postfix({'loss': f'{losses.avg(0):.4f}'})

        batch_pbar.close()

        # Apply any remaining accumulated gradients
        if num_iter > 0:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip, norm_type=2)
            optimizer.step()
            base_model.zero_grad()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        lr = optimizer.param_groups[0]['lr']

        epoch_pbar.set_postfix({'loss': f'{losses.avg(0):.4f}', 'lr': f'{lr:.2e}'})
        print_log(
            f'  E{epoch:>3d}  loss={losses.avg(0):.4f}  lr={lr:.6f}  {epoch_time:.1f}s',
            logger=logger,
        )

        # SVM validation
        if epoch % args.val_freq == 0 and epoch != 0 and extra_train_dataloader is not None:
            metrics = validate_svm(
                base_model, extra_train_dataloader, test_dataloader,
                epoch, args, config,
                npoints=config.dataset.train.others.npoints,
                return_percent=svm_return_percent,
                val_writer=val_writer, logger=logger,
            )
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(
                    base_model, optimizer, epoch, metrics, best_metrics,
                    'ckpt-best', args, logger=None,
                )

        builder.save_checkpoint(
            base_model, optimizer, epoch, metrics, best_metrics,
            'ckpt-last', args, logger=None,
        )

        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(
                base_model, optimizer, epoch, metrics, best_metrics,
                f'ckpt-epoch-{epoch:03d}', args, logger=None,
            )

    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    fmt.print_training_complete(
        best_metrics.acc * 100 if best_metrics.acc < 1 else best_metrics.acc,
        logger=logger,
    )

    return {
        'model_name': config.model.NAME,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'avg_epoch_time': avg_epoch_time,
        'best_svm_acc': best_metrics.acc,
    }


