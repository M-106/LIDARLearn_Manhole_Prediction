#!/bin/bash
# ============================================================
# SMOKE TEST: Few-shot classification (ModelNet40)
#   * 1 epoch per fold (via --max_epoch 1 forwarded to main.py)
#   * Only the 5w10s combo (smallest, fastest)
#   * Skips the 10-fold sweep — runs fold 0 only by NOT passing
#     --run_all_folds, so each model trains once on fold_0.pkl.
#
# Full few-shot sweep (10 folds × 4 combos × all models) lives
# in train_fewshot.sh; this smoke test only validates that every
# config loads + 1 train/val step completes.
# ============================================================

set -u

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-1}
COMBO=${COMBO:-5w10s}

COMMON="--mode finetune --seed $SEED --max_epoch $MAX_EPOCH --augmentation none"

# Every SSL + PEFT variant + a few supervised baselines
declare -a JOBS=(
    "dgcnn|cfgs/classification/DGCNN/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|"
    "pointnet|cfgs/classification/PointNet/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|"
    "pointnet2|cfgs/classification/PointNet2/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|"
    "pct|cfgs/classification/PCT/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|"
    "pointtransformer|cfgs/classification/PointTransformer/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|"

    # SSL — FF
    "pointmae|cfgs/classification/PointMAE/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|pretrained/pretrained_mae.pth"
    "pointbert|cfgs/classification/PointBERT/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|pretrained/pretrained_bert.pth"
    "pointgpt|cfgs/classification/PointGPT/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|pretrained/pretrained_gpt.pth"
    "act|cfgs/classification/ACT/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|pretrained/pretrained_act.pth"
    "recon|cfgs/classification/RECON/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|pretrained/pretrained_recon.pth"
    "pcpmae|cfgs/classification/PCPMAE/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|pretrained/pretrained_pcp.pth"
    "pointm2ae|cfgs/classification/PointM2AE/ModelNetFewShot/modelnet_fewshot_${COMBO}.yaml|pretrained/pretrained_m2ae.pth"

    # Full PEFT matrix for every SSL backbone that has PEFT few-shot configs
    # (5 strategies × 4 backbones = 20 PEFT jobs)
    "pointmae_dapt|cfgs/classification/PointMAE/ModelNetFewShot/modelnet_fewshot_${COMBO}_dapt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_idpt|cfgs/classification/PointMAE/ModelNetFewShot/modelnet_fewshot_${COMBO}_idpt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_ppt|cfgs/classification/PointMAE/ModelNetFewShot/modelnet_fewshot_${COMBO}_ppt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_gst|cfgs/classification/PointMAE/ModelNetFewShot/modelnet_fewshot_${COMBO}_gst.yaml|pretrained/pretrained_mae.pth"
    "pointmae_vpt_deep|cfgs/classification/PointMAE/ModelNetFewShot/modelnet_fewshot_${COMBO}_vpt_deep.yaml|pretrained/pretrained_mae.pth"

    "pointgpt_dapt|cfgs/classification/PointGPT/ModelNetFewShot/modelnet_fewshot_${COMBO}_dapt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_idpt|cfgs/classification/PointGPT/ModelNetFewShot/modelnet_fewshot_${COMBO}_idpt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_ppt|cfgs/classification/PointGPT/ModelNetFewShot/modelnet_fewshot_${COMBO}_ppt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_gst|cfgs/classification/PointGPT/ModelNetFewShot/modelnet_fewshot_${COMBO}_gst.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_vpt_deep|cfgs/classification/PointGPT/ModelNetFewShot/modelnet_fewshot_${COMBO}_vpt_deep.yaml|pretrained/pretrained_gpt.pth"

    "act_dapt|cfgs/classification/ACT/ModelNetFewShot/modelnet_fewshot_${COMBO}_dapt.yaml|pretrained/pretrained_act.pth"
    "act_idpt|cfgs/classification/ACT/ModelNetFewShot/modelnet_fewshot_${COMBO}_idpt.yaml|pretrained/pretrained_act.pth"
    "act_ppt|cfgs/classification/ACT/ModelNetFewShot/modelnet_fewshot_${COMBO}_ppt.yaml|pretrained/pretrained_act.pth"
    "act_gst|cfgs/classification/ACT/ModelNetFewShot/modelnet_fewshot_${COMBO}_gst.yaml|pretrained/pretrained_act.pth"
    "act_vpt_deep|cfgs/classification/ACT/ModelNetFewShot/modelnet_fewshot_${COMBO}_vpt_deep.yaml|pretrained/pretrained_act.pth"

    "recon_dapt|cfgs/classification/RECON/ModelNetFewShot/modelnet_fewshot_${COMBO}_dapt.yaml|pretrained/pretrained_recon.pth"
    "recon_idpt|cfgs/classification/RECON/ModelNetFewShot/modelnet_fewshot_${COMBO}_idpt.yaml|pretrained/pretrained_recon.pth"
    "recon_ppt|cfgs/classification/RECON/ModelNetFewShot/modelnet_fewshot_${COMBO}_ppt.yaml|pretrained/pretrained_recon.pth"
    "recon_gst|cfgs/classification/RECON/ModelNetFewShot/modelnet_fewshot_${COMBO}_gst.yaml|pretrained/pretrained_recon.pth"
    "recon_vpt_deep|cfgs/classification/RECON/ModelNetFewShot/modelnet_fewshot_${COMBO}_vpt_deep.yaml|pretrained/pretrained_recon.pth"
)

echo "============================================================"
echo " SMOKE: ModelNetFewShot — 1 epoch / model, combo=$COMBO"
echo " Jobs: ${#JOBS[@]}"
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
    if python main.py --config "$cfg" $COMMON $ckpt_arg --exp_name "smoke_${label}_${COMBO}"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        failures+=("$label")
        echo "  [FAIL] $label — continuing"
    fi
done

echo ""
echo "============================================================"
echo " SMOKE: FewShot finished"
echo "   ok:   $ok"
echo "   fail: $fail"
if [[ $fail -gt 0 ]]; then
    echo "   failures:"
    for f in "${failures[@]}"; do echo "     - $f"; done
fi
echo "============================================================"
