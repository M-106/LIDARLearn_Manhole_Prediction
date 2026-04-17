#!/bin/bash
# ============================================================
# SMOKE TEST: STPCTLS Cross-Validation
#   * 1 epoch per fold per model
#   * --run_all_folds enabled (validates the K-fold loop)
#
# Runs a REPRESENTATIVE SUBSET of models (not every config)
# because K_FOLDS × N_models × 1 epoch gets long. The subset
# covers the main architecture families so a break anywhere
# in the CV pipeline (dataset split, per-fold reset, aggregate
# CSV writing) is caught.
#
# To smoke-test every STPCTLS model without CV, use
# smoke_train_stpctls.sh instead.
# ============================================================

set -u

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-1}
COMMON="--mode finetune --seed $SEED --max_epoch $MAX_EPOCH --augmentation none --run_all_folds"

# Representative subset: one model per family + one SSL + one PEFT
declare -a JOBS=(
    "pointnet|cfgs/classification/PointNet/STPCTLS/stpctls.yaml|"
    "dgcnn|cfgs/classification/DGCNN/STPCTLS/stpctls.yaml|"
    "pct|cfgs/classification/PCT/STPCTLS/stpctls.yaml|"
    "pointtransformer|cfgs/classification/PointTransformer/STPCTLS/stpctls.yaml|"
    "pointmae|cfgs/classification/PointMAE/STPCTLS/stpctls.yaml|pretrained/pretrained_mae.pth"
    "pointmae_dapt|cfgs/classification/PointMAE/STPCTLS/stpctls_dapt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_idpt|cfgs/classification/PointMAE/STPCTLS/stpctls_idpt.yaml|pretrained/pretrained_mae.pth"
    "recon|cfgs/classification/RECON/STPCTLS/stpctls.yaml|pretrained/pretrained_recon.pth"
)

echo "============================================================"
echo " SMOKE: STPCTLS Cross-Validation — 1 epoch/fold"
echo " Seed: $SEED"
echo " Models: ${#JOBS[@]} (representative subset)"
echo "============================================================"

ok=0; fail=0; failures=()
for idx in "${!JOBS[@]}"; do
    IFS='|' read -r label cfg ckpt <<< "${JOBS[$idx]}"
    if [[ ! -f "$cfg" ]]; then
        echo "[$((idx+1))/${#JOBS[@]}] $label SKIP (missing: $cfg)"
        continue
    fi
    ckpt_arg=""
    if [[ -n "$ckpt" && -f "$ckpt" ]]; then
        ckpt_arg="--ckpts $ckpt"
    fi
    echo "[$((idx+1))/${#JOBS[@]}] $label"
    if python main.py --config "$cfg" $COMMON $ckpt_arg --exp_name "smoke_${label}_stpctls_cv"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        failures+=("$label")
        echo "  [FAIL] $label — continuing"
    fi
done

echo ""
echo "============================================================"
echo " SMOKE: STPCTLS-CV finished"
echo "   ok:   $ok"
echo "   fail: $fail"
if [[ $fail -gt 0 ]]; then
    echo "   failures:"
    for f in "${failures[@]}"; do echo "     - $f"; done
fi
echo "============================================================"
