"""
Base Model Class for Point Cloud Segmentation (per-point classification).

All segmentation models should inherit from BaseSegModel. The forward pass
must return per-point logits. Unlike BasePointCloudModel (which returns
[B, num_classes]), segmentation models return [B, seg_classes, N].

Author: Said Ohamouddou
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


# Milletari et al. (2016) – "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" -> hat Dice Loss für 3D Segmentierung etabliert
def dice_loss(pred, target, smooth=1.0, ignore_index=255):
    pred_flat = torch.sigmoid(pred).reshape(-1)
    target_flat = target.reshape(-1)

    # masking: only points with not 255 as label/value
    mask = target_flat != ignore_index
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask].float()

    intersection = (pred_flat * target_flat).sum()
    return 1 - (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


# Zhou et al. (2019) – "UNet++: A Nested U-Net Architecture" verwendet diese Kombination standardmäßig
def bce_dice_loss(pred, target, lambda_=1.0, ignore_index=255):
    pred_flat = pred.reshape(-1)  # Batch*N
    target_flat = target.reshape(-1)  # Batch*N

    # masking: only points with not 255 as label/value
    mask = target_flat != ignore_index
    # print(f"pred shape in bce_dice_loss: {pred.shape}")
    pred_masked = pred_flat[mask]
    target_masked = target_flat[mask].float()

    bce = F.binary_cross_entropy_with_logits(pred_masked, target_masked)
    dice = dice_loss(pred, target, ignore_index=ignore_index)
    return bce + lambda_ * dice


# Salehi et al. (2017) – "Tversky loss function for image segmentation using 3D fully convolutional deep networks", MICCAI Workshop
def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1.0, ignore_index=255):
    pred_flat = torch.sigmoid(pred).reshape(-1)
    target_flat = target.reshape(-1)

    # masking the ignore index
    mask = target_flat != ignore_index
    pred_masked = pred_flat[mask]
    target_masked = target_flat[mask].float()

    TP = (pred_masked * target_masked).sum()
    FP = ((1 - target_masked) * pred_masked).sum()
    FN = (target_masked * (1 - pred_masked)).sum()
    return 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=50.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.float32)
        )

    def forward(self, pred, target):
        target = target.float()

        # create mask FIRST
        mask = target != self.ignore_index

        # clamp target to valid range for computation
        target_valid = target.clone()
        target_valid[~mask] = 0  # avoid 255 breaking math

        # BCE (no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred,
            target_valid,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        # probabilities
        probs = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)

        # p_t
        p_t = probs * target_valid + (1 - probs) * (1 - target_valid)

        # focal term
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce_loss

        # apply mask AFTER safe computation
        loss = loss * mask.float()

        return loss.sum() / mask.sum().clamp(min=1)



def log_precision_recall(pred, probs, target, ignore_index=255, path=None):
    
    # logits = pred
    # probs = torch.sigmoid(logits)

    probs_flat = probs.reshape(-1)
    target_flat = target.reshape(-1)

    # masking the ignore index
    mask = target_flat != ignore_index
    prob_masked = probs_flat[mask]
    target_masked = target_flat[mask].float()

    pred_binary = (prob_masked > 0.5).float()

    TP = (pred_binary * target_masked).sum()
    FP = ((1 - target_masked) * pred_binary).sum()
    FN = (target_masked * (1 - pred_binary)).sum()
    
    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)

    if path is not None:
        labels = target[0]
        save_str = ""
        # ratio = labels[labels == 1].sum() / len(labels)
        ratio = (labels == 1).float().mean()
        save_str += f"Manhole ratio: {ratio:.4%}"
        save_str += f"\nUnique labels im Batch: {torch.unique(target)}"
        save_str += f"\nLabel shape: {labels.shape}, dtype: {target.dtype}"
        save_str += f"\nProbs range: {probs.min():.3f} - {probs.max():.3f}"
        save_str += f"\nPred range: {pred.min():.3f} - {pred.max():.3f}"
        save_str += f"\n\nPrecision: {precision.item()}, Recall: {recall.item()}"
        with open(path, "a") as file_:
            file_.write(save_str)

    return precision, recall



class BaseSegModel(nn.Module, ABC):
    """
    Abstract base class for point cloud segmentation models.

    Required Methods:
        - forward(pts, cls_label=None): Returns [B, seg_classes, N] log-probabilities
                                        (already log-softmax'd along dim=1).

    Notes on output convention:
        forward() returns log-probabilities (post log_softmax) with shape
        [B, seg_classes, N]. This matches the convention used by
        Point_M2AE_SEG and the PPT segmentation reference. NLL loss
        expects log-probabilities, so we use F.nll_loss in get_loss_acc.
    """

    def __init__(self, config, seg_classes=50, num_obj_classes=16, **kwargs):
        super().__init__()
        self.seg_classes = config.seg_classes
        self.num_obj_classes = config.num_obj_classes
        self.use_cls_label = config.use_cls_label
        self.config = config

    @abstractmethod
    def forward(
        self, pts: torch.Tensor, cls_label: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pts: [B, C, N] or [B, N, C] (C=3 for xyz; implementations may accept more)
            cls_label: [B, num_obj_classes] one-hot object-category label for part seg.
                       None for semantic segmentation.
        Returns:
            log-probs of shape [B, seg_classes, N]
        """
        pass

    def get_loss_acc(self, pred: torch.Tensor, gt: torch.Tensor, log_path=None) -> tuple:
        """
        Args:
            pred: [B, seg_classes, N] log-probabilities (post log_softmax)
            gt:   [B, N] long tensor of per-point labels
        Returns:
            (loss, acc) where acc is per-point accuracy as a percentage.
        """
        B, C, N = pred.shape
        # NLL loss expects [B, C, N] with class dim at position 1

        weights = torch.tensor([1.0, 1.5], device=pred.device, dtype=torch.float)

        # print(f"pred shape before loss: {pred.shape} -> will be changed to {pred[1].shape}")
        # loss = F.nll_loss(pred, gt.long(), weight=weights, ignore_index=255)
        # loss = bce_dice_loss(pred=pred[:, 1, :],  # just use probability that it is a manhole
        #                      target=gt.long(),
        #                      lambda_=1.0,
        #                      ignore_index=255)
        # loss = tversky_loss(pred[:, 1, :], gt.long(), alpha=0.3, beta=0.7, smooth=1.0)
        # loss = tversky_loss(pred[:, 1, :], gt.long(), alpha=0.05, beta=0.95, smooth=1.0)

        # loss = F.nll_loss(pred, gt.long(), weight=weights, ignore_index=255) + \
        #        tversky_loss(pred[:, 1, :], gt.long(), alpha=0.05, beta=0.95, smooth=1.0) + \
        #        bce_dice_loss(pred=pred[:, 1, :], target=gt.long(), lambda_=1.0, ignore_index=255)

        # # Pred should be Log-Softmax for NLL
        # loss_1 = F.nll_loss(pred, gt.long(), weight=weights, ignore_index=255)

        # # Convert to probabilities for Tversky (take Exp if using Log-Softmax)
        # probs = torch.exp(pred[:, 1, :]) 
        # loss_2 = tversky_loss(probs, gt.long(), alpha=0.3, beta=0.7, smooth=1.0)

        # loss = loss_1 + loss_2

        ratio = 0.015
        pos_weight = (1 - ratio) / ratio  # about 65
        logits_manhole = pred[:, 1, :]
        loss = FocalLoss(gamma=2.0, pos_weight=pos_weight, ignore_index=255)(logits_manhole, gt.long())

        # debugging
        probs = torch.sigmoid(logits_manhole)
        pred_choice_focal = (probs > 0.5).long()
        # pred_choice_argmax = pred.argmax(dim=1) 
        precision, recall = log_precision_recall(logits_manhole, probs, gt.float(), ignore_index=255, path=log_path)

        pred_choice = pred_choice_focal  # [B, N]
        
        acc = (pred_choice == gt).float().mean() * 100.0
        mask = (gt == 1) & (gt != 255)
        if mask.sum() > 0:
            manhole_acc = (pred_choice[mask] == gt[mask]).float().mean() * 100.0
        else:
            manhole_acc = torch.tensor(
                0.0, device=pred.device
            )  # if no manhole is in the batch
            # FIXME: 100 or 0 ?

        return loss, acc, manhole_acc, precision, recall

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
