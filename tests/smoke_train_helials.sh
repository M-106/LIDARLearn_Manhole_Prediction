#!/bin/bash
# ============================================================
# SMOKE TEST: HELIALS — every model / strategy for 1 epoch
#
# Mirrors train_helials.sh model list but runs each config
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
    "pointnet|cfgs/classification/PointNet/HELIALS/helials.yaml|"
    "pointnet2_ssg|cfgs/classification/PointNet2/HELIALS/helials_ssg.yaml|"
    "pointnet2_msg|cfgs/classification/PointNet2/HELIALS/helials_msg.yaml|"
    "dgcnn|cfgs/classification/DGCNN/HELIALS/helials.yaml|"
    "pct|cfgs/classification/PCT/HELIALS/helials.yaml|"
    "pointmlp|cfgs/classification/PointMLP/HELIALS/helials.yaml|"
    "curvenet|cfgs/classification/CurveNet/HELIALS/helials.yaml|"
    "deepgcn|cfgs/classification/DeepGCN/HELIALS/helials.yaml|"
    "dela|cfgs/classification/DELA/HELIALS/helials.yaml|"
    "rscnn|cfgs/classification/RSCNN/HELIALS/helials.yaml|"
    "pointconv|cfgs/classification/PointConv/HELIALS/helials.yaml|"
    "pointweb|cfgs/classification/PointWeb/HELIALS/helials.yaml|"
    "sonet|cfgs/classification/SONet/HELIALS/helials.yaml|"
    "repsurf|cfgs/classification/RepSurf/HELIALS/helials.yaml|"
    "pointcnn|cfgs/classification/PointCNN/HELIALS/helials.yaml|"
    "pointscnet|cfgs/classification/PointSCNet/HELIALS/helials.yaml|"
    "gdan|cfgs/classification/GDAN/HELIALS/helials.yaml|"
    "ppfnet|cfgs/classification/PPFNet/HELIALS/helials.yaml|"
    "pvt|cfgs/classification/PVT/HELIALS/helials.yaml|"
    "pointtransformer|cfgs/classification/PointTransformer/HELIALS/helials.yaml|"
    "pointtransformerv2|cfgs/classification/PointTransformerV2/HELIALS/helials.yaml|"
    "pointtransformerv3|cfgs/classification/PointTransformerV3/HELIALS/helials.yaml|"
    "p2p|cfgs/classification/P2P/HELIALS/helials.yaml|"
    "pointtnt|cfgs/classification/PointTNT/HELIALS/helials.yaml|"
    "globaltransformer|cfgs/classification/GlobalTransformer/HELIALS/helials.yaml|"
    "pointkan|cfgs/classification/PointKAN/HELIALS/helials.yaml|"
    "msdgcnn|cfgs/classification/MSDGCNN/HELIALS/helials.yaml|"
    "msdgcnn2|cfgs/classification/MSDGCNN2/HELIALS/helials.yaml|"
    "kandgcnn|cfgs/classification/KANDGCNN/HELIALS/helials.yaml|"

    # SSL
    "pointmae|cfgs/classification/PointMAE/HELIALS/helials.yaml|pretrained/pretrained_mae.pth"
    "pointbert|cfgs/classification/PointBERT/HELIALS/helials.yaml|pretrained/pretrained_bert.pth"
    "pointgpt|cfgs/classification/PointGPT/HELIALS/helials.yaml|pretrained/pretrained_gpt.pth"
    "act|cfgs/classification/ACT/HELIALS/helials.yaml|pretrained/pretrained_act.pth"
    "recon|cfgs/classification/RECON/HELIALS/helials.yaml|pretrained/pretrained_recon.pth"
    "pcpmae|cfgs/classification/PCPMAE/HELIALS/helials.yaml|pretrained/pretrained_pcp.pth"
    "pointm2ae|cfgs/classification/PointM2AE/HELIALS/helials.yaml|pretrained/pretrained_m2ae.pth"

    # PEFT on PointMAE
    "pointmae_idpt|cfgs/classification/PointMAE/HELIALS/helials_idpt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_vpt_deep|cfgs/classification/PointMAE/HELIALS/helials_vpt_deep.yaml|pretrained/pretrained_mae.pth"
    "pointmae_dapt|cfgs/classification/PointMAE/HELIALS/helials_dapt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_ppt|cfgs/classification/PointMAE/HELIALS/helials_ppt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_gst|cfgs/classification/PointMAE/HELIALS/helials_gst.yaml|pretrained/pretrained_mae.pth"

    # PEFT on PointGPT
    "pointgpt_idpt|cfgs/classification/PointGPT/HELIALS/helials_idpt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_vpt_deep|cfgs/classification/PointGPT/HELIALS/helials_vpt_deep.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_dapt|cfgs/classification/PointGPT/HELIALS/helials_dapt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_ppt|cfgs/classification/PointGPT/HELIALS/helials_ppt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_gst|cfgs/classification/PointGPT/HELIALS/helials_gst.yaml|pretrained/pretrained_gpt.pth"

    # PEFT on ACT
    "act_idpt|cfgs/classification/ACT/HELIALS/helials_idpt.yaml|pretrained/pretrained_act.pth"
    "act_vpt_deep|cfgs/classification/ACT/HELIALS/helials_vpt_deep.yaml|pretrained/pretrained_act.pth"
    "act_dapt|cfgs/classification/ACT/HELIALS/helials_dapt.yaml|pretrained/pretrained_act.pth"
    "act_ppt|cfgs/classification/ACT/HELIALS/helials_ppt.yaml|pretrained/pretrained_act.pth"
    "act_gst|cfgs/classification/ACT/HELIALS/helials_gst.yaml|pretrained/pretrained_act.pth"

    # PEFT on ReCon
    "recon_idpt|cfgs/classification/RECON/HELIALS/helials_idpt.yaml|pretrained/pretrained_recon.pth"
    "recon_vpt_deep|cfgs/classification/RECON/HELIALS/helials_vpt_deep.yaml|pretrained/pretrained_recon.pth"
    "recon_dapt|cfgs/classification/RECON/HELIALS/helials_dapt.yaml|pretrained/pretrained_recon.pth"
    "recon_ppt|cfgs/classification/RECON/HELIALS/helials_ppt.yaml|pretrained/pretrained_recon.pth"
    "recon_gst|cfgs/classification/RECON/HELIALS/helials_gst.yaml|pretrained/pretrained_recon.pth"
)

echo "============================================================"
echo " SMOKE: HELIALS — 1 epoch / model, seed $SEED"
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
    if python main.py --config "$cfg" $COMMON $ckpt_arg --exp_name "smoke_${label}_helials"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        failures+=("$label")
        echo "  [FAIL] $label — continuing"
    fi
done

echo ""
echo "============================================================"
echo " SMOKE: HELIALS finished"
echo "   ok:   $ok"
echo "   fail: $fail"
if [[ $fail -gt 0 ]]; then
    echo "   failures:"
    for f in "${failures[@]}"; do echo "     - $f"; done
fi
echo "============================================================"
