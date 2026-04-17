#!/bin/bash
# ============================================================
# SMOKE TEST: STPCTLS — every model / strategy for 1 epoch
#
# Mirrors train_stpctls.sh model list but runs each config
# for a single epoch to validate that everything loads and
# one train + val iteration completes without errors.
# Failures are tallied, never abort.
# ============================================================

set -u

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-1}
COMMON="--mode finetune --seed $SEED --max_epoch $MAX_EPOCH --augmentation none"

declare -a JOBS=(
    # Supervised
    "pointnet|cfgs/classification/PointNet/STPCTLS/stpctls.yaml|"
    "pointnet2_ssg|cfgs/classification/PointNet2/STPCTLS/stpctls_ssg.yaml|"
    "pointnet2_msg|cfgs/classification/PointNet2/STPCTLS/stpctls_msg.yaml|"
    "dgcnn|cfgs/classification/DGCNN/STPCTLS/stpctls.yaml|"
    "pct|cfgs/classification/PCT/STPCTLS/stpctls.yaml|"
    "pointmlp|cfgs/classification/PointMLP/STPCTLS/stpctls.yaml|"
    "curvenet|cfgs/classification/CurveNet/STPCTLS/stpctls.yaml|"
    "deepgcn|cfgs/classification/DeepGCN/STPCTLS/stpctls.yaml|"
    "dela|cfgs/classification/DELA/STPCTLS/stpctls.yaml|"
    "rscnn|cfgs/classification/RSCNN/STPCTLS/stpctls.yaml|"
    "pointconv|cfgs/classification/PointConv/STPCTLS/stpctls.yaml|"
    "pointweb|cfgs/classification/PointWeb/STPCTLS/stpctls.yaml|"
    "sonet|cfgs/classification/SONet/STPCTLS/stpctls.yaml|"
    "repsurf|cfgs/classification/RepSurf/STPCTLS/stpctls.yaml|"
    "pointcnn|cfgs/classification/PointCNN/STPCTLS/stpctls.yaml|"
    "pointscnet|cfgs/classification/PointSCNet/STPCTLS/stpctls.yaml|"
    "gdan|cfgs/classification/GDAN/STPCTLS/stpctls.yaml|"
    "ppfnet|cfgs/classification/PPFNet/STPCTLS/stpctls.yaml|"
    "pvt|cfgs/classification/PVT/STPCTLS/stpctls.yaml|"
    "pointtransformer|cfgs/classification/PointTransformer/STPCTLS/stpctls.yaml|"
    "pointtransformerv2|cfgs/classification/PointTransformerV2/STPCTLS/stpctls.yaml|"
    "pointtransformerv3|cfgs/classification/PointTransformerV3/STPCTLS/stpctls.yaml|"
    "p2p|cfgs/classification/P2P/STPCTLS/stpctls.yaml|"
    "pointtnt|cfgs/classification/PointTNT/STPCTLS/stpctls.yaml|"
    "globaltransformer|cfgs/classification/GlobalTransformer/STPCTLS/stpctls.yaml|"
    "pointkan|cfgs/classification/PointKAN/STPCTLS/stpctls.yaml|"
    "msdgcnn|cfgs/classification/MSDGCNN/STPCTLS/stpctls.yaml|"
    "msdgcnn2|cfgs/classification/MSDGCNN2/STPCTLS/stpctls.yaml|"
    "kandgcnn|cfgs/classification/KANDGCNN/STPCTLS/stpctls.yaml|"

    # SSL
    "pointmae|cfgs/classification/PointMAE/STPCTLS/stpctls.yaml|pretrained/pretrained_mae.pth"
    "pointbert|cfgs/classification/PointBERT/STPCTLS/stpctls.yaml|pretrained/pretrained_bert.pth"
    "pointgpt|cfgs/classification/PointGPT/STPCTLS/stpctls.yaml|pretrained/pretrained_gpt.pth"
    "act|cfgs/classification/ACT/STPCTLS/stpctls.yaml|pretrained/pretrained_act.pth"
    "recon|cfgs/classification/RECON/STPCTLS/stpctls.yaml|pretrained/pretrained_recon.pth"
    "pcpmae|cfgs/classification/PCPMAE/STPCTLS/stpctls.yaml|pretrained/pretrained_pcp.pth"
    "pointm2ae|cfgs/classification/PointM2AE/STPCTLS/stpctls.yaml|pretrained/pretrained_m2ae.pth"

    # PEFT on PointMAE
    "pointmae_idpt|cfgs/classification/PointMAE/STPCTLS/stpctls_idpt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_vpt_deep|cfgs/classification/PointMAE/STPCTLS/stpctls_vpt_deep.yaml|pretrained/pretrained_mae.pth"
    "pointmae_dapt|cfgs/classification/PointMAE/STPCTLS/stpctls_dapt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_ppt|cfgs/classification/PointMAE/STPCTLS/stpctls_ppt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_gst|cfgs/classification/PointMAE/STPCTLS/stpctls_gst.yaml|pretrained/pretrained_mae.pth"

    # PEFT on PointGPT
    "pointgpt_idpt|cfgs/classification/PointGPT/STPCTLS/stpctls_idpt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_vpt_deep|cfgs/classification/PointGPT/STPCTLS/stpctls_vpt_deep.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_dapt|cfgs/classification/PointGPT/STPCTLS/stpctls_dapt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_ppt|cfgs/classification/PointGPT/STPCTLS/stpctls_ppt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_gst|cfgs/classification/PointGPT/STPCTLS/stpctls_gst.yaml|pretrained/pretrained_gpt.pth"

    # PEFT on ACT
    "act_idpt|cfgs/classification/ACT/STPCTLS/stpctls_idpt.yaml|pretrained/pretrained_act.pth"
    "act_vpt_deep|cfgs/classification/ACT/STPCTLS/stpctls_vpt_deep.yaml|pretrained/pretrained_act.pth"
    "act_dapt|cfgs/classification/ACT/STPCTLS/stpctls_dapt.yaml|pretrained/pretrained_act.pth"
    "act_ppt|cfgs/classification/ACT/STPCTLS/stpctls_ppt.yaml|pretrained/pretrained_act.pth"
    "act_gst|cfgs/classification/ACT/STPCTLS/stpctls_gst.yaml|pretrained/pretrained_act.pth"

    # PEFT on ReCon
    "recon_idpt|cfgs/classification/RECON/STPCTLS/stpctls_idpt.yaml|pretrained/pretrained_recon.pth"
    "recon_vpt_deep|cfgs/classification/RECON/STPCTLS/stpctls_vpt_deep.yaml|pretrained/pretrained_recon.pth"
    "recon_dapt|cfgs/classification/RECON/STPCTLS/stpctls_dapt.yaml|pretrained/pretrained_recon.pth"
    "recon_ppt|cfgs/classification/RECON/STPCTLS/stpctls_ppt.yaml|pretrained/pretrained_recon.pth"
    "recon_gst|cfgs/classification/RECON/STPCTLS/stpctls_gst.yaml|pretrained/pretrained_recon.pth"
)

echo "============================================================"
echo " SMOKE: STPCTLS — 1 epoch / model, seed $SEED"
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
    if python main.py --config "$cfg" $COMMON $ckpt_arg --exp_name "smoke_${label}_stpctls"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        failures+=("$label")
        echo "  [FAIL] $label — continuing"
    fi
done

echo ""
echo "============================================================"
echo " SMOKE: STPCTLS finished"
echo "   ok:   $ok"
echo "   fail: $fail"
if [[ $fail -gt 0 ]]; then
    echo "   failures:"
    for f in "${failures[@]}"; do echo "     - $f"; done
fi
echo "============================================================"
