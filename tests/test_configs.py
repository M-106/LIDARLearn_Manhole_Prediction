"""
L1 tests: every YAML config parses and its model builds.

No dataset loading, no forward pass, no CUDA. Just:
    yaml  ─parse─▶  EasyDict  ─build─▶  nn.Module (on CPU)

Runs in well under a minute for 500+ configs. Catches:
  * yaml syntax errors / missing _base_ paths
  * ``model.NAME`` not registered (bad import in models/__init__.py)
  * Missing required config fields (model __init__ AttributeError)
  * Half-written config files
"""

from __future__ import annotations

import pytest

from utils.config import cfg_from_yaml_file
from models.build import MODELS, build_model_from_cfg

from .conftest import (
    ALL_CONFIGS,
    CLASSIFICATION_CONFIGS,
    S3DIS_CONFIGS,
    SHAPENETPARTS_CONFIGS,
    config_id,
)


# --------------------------------------------------------------------------
# Parse step — runs on every single yaml under cfgs/
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_path", ALL_CONFIGS, ids=config_id)
def test_yaml_parses(cfg_path):
    """Every yaml under cfgs/ must be loadable by the library's config loader."""
    cfg = cfg_from_yaml_file(cfg_path)
    assert cfg is not None


# --------------------------------------------------------------------------
# Build step — one test per classification / seg / partseg config
# --------------------------------------------------------------------------
def _assert_registered(name: str):
    """Fail with a helpful message if the model name isn't in the registry.

    The registry class in ``utils/registry.py`` stores its entries in
    ``_module_dict`` and implements ``__contains__``, so membership can be
    checked via ``name in MODELS``. We use the explicit dict keys for the
    error message so the user sees what IS registered.
    """
    if name in MODELS:
        return
    try:
        registered = sorted(MODELS._module_dict.keys())
    except AttributeError:
        registered = []
    hint = ", ".join(registered[:20]) + ("..." if len(registered) > 20 else "")
    raise AssertionError(
        f"Model '{name}' is not registered "
        f"(registry has {len(registered)} entries: {hint}). "
        f"Check models/__init__.py for a missing import, "
        f"or the class's @MODELS.register_module() decorator."
    )


@pytest.mark.parametrize("cfg_path", CLASSIFICATION_CONFIGS, ids=config_id)
def test_classification_model_builds(cfg_path):
    cfg = cfg_from_yaml_file(cfg_path)
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "NAME"):
        pytest.skip("not a model config (no model.NAME)")
    _assert_registered(cfg.model.NAME)
    model = build_model_from_cfg(cfg.model)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0, f"Model {cfg.model.NAME} has no parameters"


@pytest.mark.parametrize("cfg_path", S3DIS_CONFIGS, ids=config_id)
def test_semantic_seg_model_builds(cfg_path):
    cfg = cfg_from_yaml_file(cfg_path)
    _assert_registered(cfg.model.NAME)
    model = build_model_from_cfg(cfg.model)
    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.parametrize("cfg_path", SHAPENETPARTS_CONFIGS, ids=config_id)
def test_part_seg_model_builds(cfg_path):
    cfg = cfg_from_yaml_file(cfg_path)
    _assert_registered(cfg.model.NAME)
    model = build_model_from_cfg(cfg.model)
    assert sum(p.numel() for p in model.parameters()) > 0
