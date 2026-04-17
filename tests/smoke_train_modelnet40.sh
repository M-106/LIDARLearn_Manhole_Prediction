#!/bin/bash
# ============================================================
# SMOKE TEST: ModelNet40 — every model / strategy for 1 epoch
#
# Proves every config under cfgs/classification/*/ModelNet40/
# loads + forward/backward passes + saves. Uses the same model
# registry as train_modelnet40.sh. Takes ~10-20 minutes total
# on a single GPU (1 epoch each, full data).
#
# Failures are tallied and reported at the end; a crashing
# config does not abort the sweep.
# ============================================================

set -u

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-1}
COMMON="--mode finetune --seed $SEED --max_epoch $MAX_EPOCH --augmentation none"

# (label, config, optional_ckpt)
declare -a JOBS=(
    # Supervised
    "pointnet|cfgs/classification/PointNet/ModelNet40/modelnet40.yaml|"
    "pointnet2_ssg|cfgs/classification/PointNet2/ModelNet40/modelnet40_ssg.yaml|"
    "pointnet2_msg|cfgs/classification/PointNet2/ModelNet40/modelnet40_msg.yaml|"
    "dgcnn_k20|cfgs/classification/DGCNN/ModelNet40/modelnet40_k20.yaml|"
    "msdgcnn|cfgs/classification/MSDGCNN/ModelNet40/modelnet40.yaml|"
    "msdgcnn2|cfgs/classification/MSDGCNN2/ModelNet40/modelnet40.yaml|"
    "pct|cfgs/classification/PCT/ModelNet40/modelnet40.yaml|"
    "pointmlp|cfgs/classification/PointMLP/ModelNet40/modelnet40.yaml|"
    "curvenet|cfgs/classification/CurveNet/ModelNet40/modelnet40.yaml|"
    "deepgcn|cfgs/classification/DeepGCN/ModelNet40/modelnet40.yaml|"
    "dela|cfgs/classification/DELA/ModelNet40/modelnet40.yaml|"
    "rscnn|cfgs/classification/RSCNN/ModelNet40/modelnet40.yaml|"
    "pointconv|cfgs/classification/PointConv/ModelNet40/modelnet40.yaml|"
    "pointweb|cfgs/classification/PointWeb/ModelNet40/modelnet40.yaml|"
    "sonet|cfgs/classification/SONet/ModelNet40/modelnet40.yaml|"
    "repsurf|cfgs/classification/RepSurf/ModelNet40/modelnet40.yaml|"
    "pointcnn|cfgs/classification/PointCNN/ModelNet40/modelnet40.yaml|"
    "pointscnet|cfgs/classification/PointSCNet/ModelNet40/modelnet40.yaml|"
    "gdan|cfgs/classification/GDAN/ModelNet40/modelnet40.yaml|"
    "ppfnet|cfgs/classification/PPFNet/ModelNet40/modelnet40.yaml|"
    "pvt|cfgs/classification/PVT/ModelNet40/modelnet40.yaml|"
    "pointtransformer|cfgs/classification/PointTransformer/ModelNet40/modelnet40.yaml|"
    "pointtransformerv2|cfgs/classification/PointTransformerV2/ModelNet40/modelnet40.yaml|"
    "pointtransformerv3|cfgs/classification/PointTransformerV3/ModelNet40/modelnet40.yaml|"
    "p2p|cfgs/classification/P2P/ModelNet40/modelnet40.yaml|"
    "pointtnt|cfgs/classification/PointTNT/ModelNet40/modelnet40.yaml|"
    "globaltransformer|cfgs/classification/GlobalTransformer/ModelNet40/modelnet40.yaml|"
    "pointkan|cfgs/classification/PointKAN/ModelNet40/modelnet40.yaml|"
    "kandgcnn|cfgs/classification/KANDGCNN/ModelNet40/modelnet40.yaml|"

    # SSL — Full Finetuning (with pretrained ckpt when present)
    "pointmae|cfgs/classification/PointMAE/ModelNet40/modelnet40.yaml|pretrained/pretrained_mae.pth"
    "recon|cfgs/classification/RECON/ModelNet40/modelnet40.yaml|pretrained/pretrained_recon.pth"
    "pointm2ae|cfgs/classification/PointM2AE/ModelNet40/modelnet40.yaml|pretrained/pretrained_m2ae.pth"
    "pointbert|cfgs/classification/PointBERT/ModelNet40/modelnet40.yaml|pretrained/pretrained_bert.pth"
    "pointgpt|cfgs/classification/PointGPT/ModelNet40/modelnet40.yaml|pretrained/pretrained_gpt.pth"
    "act|cfgs/classification/ACT/ModelNet40/modelnet40.yaml|pretrained/pretrained_act.pth"
    "pcpmae|cfgs/classification/PCPMAE/ModelNet40/modelnet40.yaml|pretrained/pretrained_pcp.pth"

    # PEFT on PointMAE
    "pointmae_idpt|cfgs/classification/PointMAE/ModelNet40/modelnet40_idpt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_vpt_deep|cfgs/classification/PointMAE/ModelNet40/modelnet40_vpt_deep.yaml|pretrained/pretrained_mae.pth"
    "pointmae_dapt|cfgs/classification/PointMAE/ModelNet40/modelnet40_dapt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_ppt|cfgs/classification/PointMAE/ModelNet40/modelnet40_ppt.yaml|pretrained/pretrained_mae.pth"
    "pointmae_gst|cfgs/classification/PointMAE/ModelNet40/modelnet40_gst.yaml|pretrained/pretrained_mae.pth"

    # PEFT on ReCon
    "recon_idpt|cfgs/classification/RECON/ModelNet40/modelnet40_idpt.yaml|pretrained/pretrained_recon.pth"
    "recon_vpt_deep|cfgs/classification/RECON/ModelNet40/modelnet40_vpt_deep.yaml|pretrained/pretrained_recon.pth"
    "recon_dapt|cfgs/classification/RECON/ModelNet40/modelnet40_dapt.yaml|pretrained/pretrained_recon.pth"
    "recon_ppt|cfgs/classification/RECON/ModelNet40/modelnet40_ppt.yaml|pretrained/pretrained_recon.pth"
    "recon_gst|cfgs/classification/RECON/ModelNet40/modelnet40_gst.yaml|pretrained/pretrained_recon.pth"

    # PEFT on ACT
    "act_idpt|cfgs/classification/ACT/ModelNet40/modelnet40_idpt.yaml|pretrained/pretrained_act.pth"
    "act_vpt_deep|cfgs/classification/ACT/ModelNet40/modelnet40_vpt_deep.yaml|pretrained/pretrained_act.pth"
    "act_dapt|cfgs/classification/ACT/ModelNet40/modelnet40_dapt.yaml|pretrained/pretrained_act.pth"
    "act_ppt|cfgs/classification/ACT/ModelNet40/modelnet40_ppt.yaml|pretrained/pretrained_act.pth"
    "act_gst|cfgs/classification/ACT/ModelNet40/modelnet40_gst.yaml|pretrained/pretrained_act.pth"

    # PEFT on PointGPT
    "pointgpt_idpt|cfgs/classification/PointGPT/ModelNet40/modelnet40_idpt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_vpt_deep|cfgs/classification/PointGPT/ModelNet40/modelnet40_vpt_deep.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_dapt|cfgs/classification/PointGPT/ModelNet40/modelnet40_dapt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_ppt|cfgs/classification/PointGPT/ModelNet40/modelnet40_ppt.yaml|pretrained/pretrained_gpt.pth"
    "pointgpt_gst|cfgs/classification/PointGPT/ModelNet40/modelnet40_gst.yaml|pretrained/pretrained_gpt.pth"
)

echo "============================================================"
echo " SMOKE: ModelNet40 — 1 epoch / model, seed $SEED"
echo " Jobs: ${#JOBS[@]}"
echo "============================================================"

ok=0; fail=0; failures=()
for idx in "${!JOBS[@]}"; do
    IFS='|' read -r label cfg ckpt <<< "${JOBS[$idx]}"
    if [[ ! -f "$cfg" ]]; then
        echo "[$((idx+1))/${#JOBS[@]}] $label SKIP (missing config: $cfg)"
        continue
    fi
    ckpt_arg=""
    if [[ -n "$ckpt" && -f "$ckpt" ]]; then
        ckpt_arg="--ckpts $ckpt"
    fi
    echo "[$((idx+1))/${#JOBS[@]}] $label"
    if python main.py --config "$cfg" $COMMON $ckpt_arg --exp_name "smoke_${label}_modelnet40"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        failures+=("$label")
        echo "  [FAIL] $label — continuing"
    fi
done

echo ""
echo "============================================================"
echo " SMOKE: ModelNet40 finished"
echo "   ok:   $ok"
echo "   fail: $fail"
if [[ $fail -gt 0 ]]; then
    echo "   failures:"
    for f in "${failures[@]}"; do echo "     - $f"; done
fi
echo "============================================================"
