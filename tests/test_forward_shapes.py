"""
L2 tests: forward pass + output shape verification.

Covers three tasks with distinct input/output conventions:

  Classification
      input  : (B, N, 3)
      call   : model(points)
      output : (B, num_classes)  — may be wrapped in a (logits, aux) tuple

  Semantic segmentation (S3DIS)
      input  : (B, C, N)  channel-first
      call   : model(points_bcn, None)
      output : (B, num_seg_classes, N) or (B, N, num_seg_classes)

  Part segmentation (ShapeNetParts)
      input  : (B, C, N)  channel-first + one-hot category label (B, n_obj)
      call   : model(points_bcn, cls_onehot)
      output : (B, num_seg_classes, N) or (B, N, num_seg_classes)

Every test uses a tiny random batch (B=2, N=1024) so the suite runs in a
couple of minutes on GPU without touching real data.
"""

from __future__ import annotations

import pytest
import torch

from utils.config import cfg_from_yaml_file
from models.build import build_model_from_cfg

from .conftest import (
    BATCH_SIZE,
    NUM_POINTS,
    CLASSIFICATION_NONPRETRAIN,
    S3DIS_CONFIGS,
    SHAPENETPARTS_CONFIGS,
    config_id,
    expected_cls_classes,
    expected_seg_classes,
    infer_seg_channels,
    make_cls_batch,
    make_cls_label,
    make_seg_batch,
    num_obj_classes,
    unwrap,
)

pytestmark = pytest.mark.gpu  # forward passes need CUDA for most of these


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _build_on_gpu(cfg_path):
    cfg = cfg_from_yaml_file(cfg_path)
    model = build_model_from_cfg(cfg.model).cuda().eval()
    return cfg, model


def _assert_finite(t: torch.Tensor, label: str):
    assert torch.isfinite(t).all(), f"{label}: non-finite values in output"


def _check_seg_shape(out: torch.Tensor, expected_classes: int, n: int, label: str):
    """Seg models differ on whether channels are axis 1 or axis 2.
    Accept both ``(B, C, N)`` and ``(B, N, C)``."""
    assert out.dim() == 3, f"{label}: expected 3D output, got {tuple(out.shape)}"
    assert out.shape[0] == BATCH_SIZE, \
        f"{label}: batch dim {out.shape[0]} != {BATCH_SIZE}"

    if out.shape[1] == expected_classes and out.shape[2] == n:
        return  # (B, C, N) layout
    if out.shape[2] == expected_classes and out.shape[1] == n:
        return  # (B, N, C) layout
    raise AssertionError(
        f"{label}: output shape {tuple(out.shape)} "
        f"does not match either (B={BATCH_SIZE}, C={expected_classes}, N={n}) "
        f"or (B={BATCH_SIZE}, N={n}, C={expected_classes})"
    )


# --------------------------------------------------------------------------
# Classification forward pass
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_path", CLASSIFICATION_NONPRETRAIN, ids=config_id)
def test_classification_forward_shape(cfg_path):
    """Every classification model must return ``(B, num_classes)`` logits."""
    cfg, model = _build_on_gpu(cfg_path)
    expected = expected_cls_classes(cfg)

    n = int(cfg.get("npoints", NUM_POINTS)) if hasattr(cfg, "get") else NUM_POINTS
    points = make_cls_batch(channels=3, n=n, device="cuda")

    with torch.no_grad():
        out = unwrap(model(points))

    assert out.dim() == 2, (
        f"{cfg.model.NAME}: classification output must be 2D (B, C), "
        f"got {tuple(out.shape)}"
    )
    assert out.shape == (BATCH_SIZE, expected), (
        f"{cfg.model.NAME}: classification output shape {tuple(out.shape)} "
        f"!= expected ({BATCH_SIZE}, {expected})"
    )
    _assert_finite(out, cfg.model.NAME)


# --------------------------------------------------------------------------
# Semantic segmentation (S3DIS)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_path", S3DIS_CONFIGS, ids=config_id)
def test_semantic_seg_forward_shape(cfg_path):
    """S3DIS seg models must return per-point logits of size ``seg_classes``."""
    cfg, model = _build_on_gpu(cfg_path)
    expected = expected_seg_classes(cfg)
    channels = infer_seg_channels(cfg)
    n = int(cfg.get("npoints", NUM_POINTS)) if hasattr(cfg, "get") else NUM_POINTS

    points_bcn = make_seg_batch(channels=channels, n=n, device="cuda")

    with torch.no_grad():
        out = unwrap(model(points_bcn, None))  # no cls label for semantic seg

    _check_seg_shape(out, expected_classes=expected, n=n,
                     label=f"{cfg.model.NAME} (S3DIS)")
    _assert_finite(out, cfg.model.NAME)


# --------------------------------------------------------------------------
# Part segmentation (ShapeNetParts)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_path", SHAPENETPARTS_CONFIGS, ids=config_id)
def test_part_seg_forward_shape(cfg_path):
    """ShapeNetParts models take (points_bcn, cls_onehot) and produce per-point
    logits of size ``seg_classes`` (=50 for the 50 part labels)."""
    cfg, model = _build_on_gpu(cfg_path)
    expected = expected_seg_classes(cfg)
    n_obj = num_obj_classes(cfg)
    n = int(cfg.get("npoints", NUM_POINTS)) if hasattr(cfg, "get") else NUM_POINTS

    points_bcn = make_seg_batch(channels=3, n=n, device="cuda")
    cls_onehot = make_cls_label(n_obj, device="cuda")

    with torch.no_grad():
        if cfg.model.get("use_cls_label", True):
            out = unwrap(model(points_bcn, cls_onehot))
        else:
            out = unwrap(model(points_bcn, None))

    _check_seg_shape(out, expected_classes=expected, n=n,
                     label=f"{cfg.model.NAME} (ShapeNetParts)")
    _assert_finite(out, cfg.model.NAME)


# --------------------------------------------------------------------------
# Sanity: batch dimension is respected (catches hardcoded-batch bugs)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cfg_path",
    # One representative per category to keep this test cheap
    [
        "cfgs/classification/PointNet/ModelNet40/modelnet40.yaml",
        "cfgs/classification/DGCNN/ModelNet40/modelnet40_k20.yaml",
        "cfgs/classification/PointMAE/ModelNet40/modelnet40.yaml",
    ],
    ids=lambda p: p.split("/")[-1],
)
def test_classification_batch_dim_independence(cfg_path):
    """Running with B=1 and B=3 must both work and scale linearly.

    Catches models that hardcode the batch size via buffer shapes."""
    from pathlib import Path
    from .conftest import REPO_ROOT  # type: ignore  # REPO_ROOT lives in conftest

    path = REPO_ROOT / cfg_path if not Path(cfg_path).is_absolute() else Path(cfg_path)
    cfg = cfg_from_yaml_file(path)
    model = build_model_from_cfg(cfg.model).cuda().eval()
    expected = expected_cls_classes(cfg)

    for B in (1, 3):
        torch.manual_seed(B)
        x = torch.randn(B, NUM_POINTS, 3, device="cuda")
        with torch.no_grad():
            out = unwrap(model(x))
        assert out.shape == (B, expected), (
            f"{cfg.model.NAME}: B={B} input produced shape {tuple(out.shape)} "
            f"(expected ({B}, {expected}))"
        )
