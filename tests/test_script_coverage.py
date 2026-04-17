"""
Meta-test: every classification / segmentation config under ``cfgs/`` must
be referenced by at least one shell script under ``scripts/``.

This catches "silent drift" between configs and training scripts — the
exact kind of bug where a new ``act_gst.yaml`` is added but the sweep
scripts (which hand-list every config) never pick it up, so the config
exists and builds but is never actually trained end-to-end.

The other test files already prove every config *parses and builds*
(``test_configs.py``) and *forward-passes to the right shape*
(``test_forward_shapes.py``). This test proves every config is
*actually invoked by the training pipeline*.

Rules
-----
* Pretrain configs (``*_pretrain.yaml``) are exempt — they're run via
  a different code path (``--mode pretrain``) and sometimes not wired
  into any sweep script yet.
* Dataset base yamls under ``cfgs/dataset/`` are exempt — they're only
  referenced via ``_base_`` from other yamls, not from scripts.
* A script is considered to "reference" a config if the config's relative
  path (from repo root) appears anywhere in the script text. This is a
  substring match, so both direct invocations and loop-array entries
  are picked up.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .conftest import REPO_ROOT  # type: ignore


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _discover_configs():
    """Return the set of yaml files that must be referenced by some script."""
    yamls = []
    yamls.extend(REPO_ROOT.glob("cfgs/classification/**/*.yaml"))
    yamls.extend(REPO_ROOT.glob("cfgs/segmentation/**/*.yaml"))

    # Skip pretrain configs — they belong to a different runner and are
    # often intentionally unreferenced until a pretrain campaign is started.
    yamls = [p for p in yamls if "_pretrain" not in p.name]

    return sorted(yamls)


def _discover_scripts_text():
    """Load the text of every training shell script once.

    Returns ``{script_name: content}`` so failure messages can tell the
    user which script(s) cover a given config (empty set = orphan)."""
    scripts = {}
    for sh in REPO_ROOT.glob("scripts/train_*.sh"):
        scripts[sh.name] = sh.read_text()
    for sh in REPO_ROOT.glob("scripts/smoke_train_*.sh"):
        scripts[sh.name] = sh.read_text()
    return scripts


ALL_CONFIGS = _discover_configs()
ALL_SCRIPTS = _discover_scripts_text()


def config_id(path):
    return str(path.relative_to(REPO_ROOT))


_FEWSHOT_FILENAME_RE = re.compile(
    r"^modelnet_fewshot_(?P<combo>\d+w\d+s)(?:_(?P<strategy>[a-z_]+))?\.yaml$"
)


def _fewshot_dynamic_coverage(cfg_path: Path) -> list[str]:
    """Special-case cover-detection for ``ModelNetFewShot`` configs.

    ``scripts/train_fewshot.sh`` does not hard-list individual yaml paths
    — it iterates ``CFG_DIR × combos × strategies`` and builds the path
    at runtime. The substring matcher in ``_covering_scripts`` can't see
    that, so we reimplement the script's dispatch logic here:

      * Extract ``<SubDir>`` (e.g. ``RSCNN``) and ``<combo>``/``<strategy>``
        from the config filename.
      * A config is covered by ``train_fewshot.sh`` iff its ``SubDir``
        appears on the right-hand side of a ``[key]=SubDir`` entry in the
        ``declare -A CFG_DIR`` block, AND its combo is in the default
        combo list, AND its strategy is in the default strategy list.
      * ``smoke_train_fewshot.sh`` expands ``${COMBO}`` in a hard-coded
        jobs array. We substring-match on the template form
        ``cfgs/classification/<Sub>/ModelNetFewShot/modelnet_fewshot_${COMBO}[_<strat>].yaml``.

    Returns the names of scripts that cover this config, or ``[]``.
    """
    parts = cfg_path.relative_to(REPO_ROOT).parts
    # Expected shape: cfgs / classification / <SubDir> / ModelNetFewShot / modelnet_fewshot_*.yaml
    if len(parts) != 5 or parts[0] != "cfgs" or parts[1] != "classification":
        return []
    if parts[3] != "ModelNetFewShot":
        return []
    sub_dir = parts[2]
    match = _FEWSHOT_FILENAME_RE.match(parts[4])
    if not match:
        return []
    combo = match.group("combo")
    strategy = match.group("strategy") or "ff"

    covering = []

    # ------------------------------------------------------------
    # train_fewshot.sh — dynamic CFG_DIR × combos × strategies sweep
    # ------------------------------------------------------------
    tf = ALL_SCRIPTS.get("train_fewshot.sh", "")
    if tf:
        # Check that sub_dir is enlisted in CFG_DIR (any line with ]=SubDir)
        in_cfg_dir = re.search(
            rf"\]\s*=\s*{re.escape(sub_dir)}\b", tf
        )
        # Check that the combo is in COMBOS_DEFAULT
        combos_match = re.search(r'COMBOS_DEFAULT="([^"]*)"', tf)
        combos_ok = combos_match and combo in combos_match.group(1).split()
        # Check that the strategy is in STRATEGIES_DEFAULT
        strategies_match = re.search(r'STRATEGIES_DEFAULT="([^"]*)"', tf)
        strategies_ok = strategies_match and strategy in strategies_match.group(1).split()
        if in_cfg_dir and combos_ok and strategies_ok:
            covering.append("train_fewshot.sh")

    # ------------------------------------------------------------
    # smoke_train_fewshot.sh — hard-coded jobs array with ${COMBO}
    # ------------------------------------------------------------
    sf = ALL_SCRIPTS.get("smoke_train_fewshot.sh", "")
    if sf:
        suffix = f"_{strategy}.yaml" if strategy != "ff" else ".yaml"
        template = (
            f"cfgs/classification/{sub_dir}/ModelNetFewShot/"
            f"modelnet_fewshot_${{COMBO}}{suffix}"
        )
        if template in sf:
            covering.append("smoke_train_fewshot.sh")

    return covering


def _covering_scripts(cfg_path: Path) -> list[str]:
    """Return the names of scripts that reference this config path.

    First tries a plain substring match against the literal relative path
    (catches the 95% case: scripts that hand-list every yaml). Falls back
    to :func:`_fewshot_dynamic_coverage` for ``ModelNetFewShot`` configs
    that are only reachable via ``train_fewshot.sh``'s dynamic sweep.
    """
    rel = str(cfg_path.relative_to(REPO_ROOT))
    literal = [name for name, body in ALL_SCRIPTS.items() if rel in body]
    if literal:
        return literal
    return _fewshot_dynamic_coverage(cfg_path)


# ---------------------------------------------------------------------------
# Per-config coverage test (fine-grained: one test per config)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_path", ALL_CONFIGS, ids=config_id)
def test_config_is_referenced_by_some_script(cfg_path):
    """Every non-pretrain classification/segmentation config must be run
    by at least one training or smoke script."""
    covering = _covering_scripts(cfg_path)
    assert covering, (
        f"Config {cfg_path.relative_to(REPO_ROOT)} is not invoked by any "
        f"training script under scripts/. Either add it to a train_*.sh / "
        f"smoke_train_*.sh sweep, or delete the config if it's obsolete. "
        f"(Available scripts: {sorted(ALL_SCRIPTS.keys())})"
    )


# ---------------------------------------------------------------------------
# Summary test (coarse: one test, lists all orphans at once)
# ---------------------------------------------------------------------------
def test_no_orphan_configs_summary():
    """Same check in aggregate form — useful when running the suite
    headlessly and wanting one clean failure message listing every
    orphaned config instead of N individual failures."""
    orphans = [
        str(p.relative_to(REPO_ROOT))
        for p in ALL_CONFIGS
        if not _covering_scripts(p)
    ]
    assert not orphans, (
        f"{len(orphans)} config(s) are not referenced by any training "
        f"script:\n  " + "\n  ".join(orphans)
    )
