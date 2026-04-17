#!/bin/bash
# ============================================================
# SMOKE TEST: ShapeNetParts part segmentation
#   * 1 epoch per model
#   * --data_fraction 0.05
# ============================================================

set -u

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-1}
DATA_FRACTION=${DATA_FRACTION:-0.05}

COMMON="--mode seg --seed $SEED --max_epoch $MAX_EPOCH --data_fraction $DATA_FRACTION"

declare -a JOBS=(
    # Supervised
    "pointnet|cfgs/segmentation/PointNet/ShapeNetParts/pointnet_partseg.yaml|"
    "dgcnn|cfgs/segmentation/DGCNN/ShapeNetParts/dgcnn_partseg.yaml|"
    "msdgcnn|cfgs/segmentation/MSDGCNN/ShapeNetParts/msdgcnn_partseg.yaml|"
    "msdgcnn2|cfgs/segmentation/MSDGCNN2/ShapeNetParts/msdgcnn2_partseg.yaml|"
    "gdan|cfgs/segmentation/GDAN/ShapeNetParts/gdan_partseg.yaml|"
    "kandgcnn|cfgs/segmentation/KANDGCNN/ShapeNetParts/kandgcnn_partseg.yaml|"
    "pointkan|cfgs/segmentation/PointKAN/ShapeNetParts/pointkan_partseg.yaml|"
    "deepgcn|cfgs/segmentation/DeepGCN/ShapeNetParts/deepgcn_partseg.yaml|"
    "pointnet2|cfgs/segmentation/PointNet2/ShapeNetParts/pointnet2_partseg.yaml|"
    "curvenet|cfgs/segmentation/CurveNet/ShapeNetParts/curvenet_partseg.yaml|"
    "dela|cfgs/segmentation/DELA/ShapeNetParts/dela_partseg.yaml|"
    "pct|cfgs/segmentation/PCT/ShapeNetParts/pct_partseg.yaml|"
    "pointscnet|cfgs/segmentation/PointSCNet/ShapeNetParts/pointscnet_partseg.yaml|"
    "rscnn|cfgs/segmentation/RSCNN/ShapeNetParts/rscnn_partseg.yaml|"
    "pointconv|cfgs/segmentation/PointConv/ShapeNetParts/pointconv_partseg.yaml|"
    "pvt|cfgs/segmentation/PVT/ShapeNetParts/pvt_partseg.yaml|"
    "pointweb|cfgs/segmentation/PointWeb/ShapeNetParts/pointweb_partseg.yaml|"
    "pointmlp|cfgs/segmentation/PointMLP/ShapeNetParts/pointmlp_partseg.yaml|"
    "randlanet|cfgs/segmentation/RandLANet/ShapeNetParts/randlanet_partseg.yaml|"
    "repsurf|cfgs/segmentation/RepSurf/ShapeNetParts/repsurf_partseg.yaml|"
    "pointtransformer|cfgs/segmentation/PointTransformer/ShapeNetParts/pointtransformer_partseg.yaml|"
    "pointtransformerv2|cfgs/segmentation/PointTransformerV2/ShapeNetParts/pointtransformerv2_partseg.yaml|"
    "pointtransformerv3|cfgs/segmentation/PointTransformerV3/ShapeNetParts/pointtransformerv3_partseg.yaml|"
    "p2p|cfgs/segmentation/P2P/ShapeNetParts/p2p_partseg.yaml|"
    "globaltransformer|cfgs/segmentation/GlobalTransformer/ShapeNetParts/globaltransformer_partseg.yaml|"
    "pointtnt|cfgs/segmentation/PointTNT/ShapeNetParts/pointtnt_partseg.yaml|"

    # SSL
    "pointmae|cfgs/segmentation/PointMAE/ShapeNetParts/pointmae_partseg.yaml|pretrained/pretrained_mae.pth"
    "pointbert|cfgs/segmentation/PointBERT/ShapeNetParts/pointbert_partseg.yaml|pretrained/pretrained_bert.pth"
    "pointgpt|cfgs/segmentation/PointGPT/ShapeNetParts/pointgpt_partseg.yaml|pretrained/pretrained_gpt.pth"
    "act|cfgs/segmentation/ACT/ShapeNetParts/act_partseg.yaml|pretrained/pretrained_act.pth"
    "recon|cfgs/segmentation/RECON/ShapeNetParts/recon_partseg.yaml|pretrained/pretrained_recon.pth"
    "pcpmae|cfgs/segmentation/PCPMAE/ShapeNetParts/pcpmae_partseg.yaml|pretrained/pretrained_pcp.pth"
    "pointm2ae|cfgs/segmentation/PointM2AE/ShapeNetParts/pointm2ae_partseg.yaml|pretrained/pretrained_m2ae.pth"

    # PEFT
    "pointmae_ppt|cfgs/segmentation/PointMAE/ShapeNetParts/pointmae_partseg_ppt.yaml|pretrained/pretrained_mae.pth"
    "act_ppt|cfgs/segmentation/ACT/ShapeNetParts/act_partseg_ppt.yaml|pretrained/pretrained_act.pth"
    "recon_ppt|cfgs/segmentation/RECON/ShapeNetParts/recon_partseg_ppt.yaml|pretrained/pretrained_recon.pth"
    "pointmae_dapt|cfgs/segmentation/PointMAE/ShapeNetParts/pointmae_partseg_dapt.yaml|pretrained/pretrained_mae.pth"
    "act_dapt|cfgs/segmentation/ACT/ShapeNetParts/act_partseg_dapt.yaml|pretrained/pretrained_act.pth"
    "recon_dapt|cfgs/segmentation/RECON/ShapeNetParts/recon_partseg_dapt.yaml|pretrained/pretrained_recon.pth"
    "pointmae_idpt|cfgs/segmentation/PointMAE/ShapeNetParts/pointmae_partseg_idpt.yaml|pretrained/pretrained_mae.pth"
    "act_idpt|cfgs/segmentation/ACT/ShapeNetParts/act_partseg_idpt.yaml|pretrained/pretrained_act.pth"
    "recon_idpt|cfgs/segmentation/RECON/ShapeNetParts/recon_partseg_idpt.yaml|pretrained/pretrained_recon.pth"
    "pointmae_gst|cfgs/segmentation/PointMAE/ShapeNetParts/pointmae_partseg_gst.yaml|pretrained/pretrained_mae.pth"
    "act_gst|cfgs/segmentation/ACT/ShapeNetParts/act_partseg_gst.yaml|pretrained/pretrained_act.pth"
    "recon_gst|cfgs/segmentation/RECON/ShapeNetParts/recon_partseg_gst.yaml|pretrained/pretrained_recon.pth"
)

echo "============================================================"
echo " SMOKE: ShapeNetParts — 1 epoch / model, data_fraction=$DATA_FRACTION"
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
    if python main.py --config "$cfg" $COMMON $ckpt_arg --exp_name "smoke_${label}_partseg"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        failures+=("$label")
        echo "  [FAIL] $label — continuing"
    fi
done

echo ""
echo "============================================================"
echo " SMOKE: ShapeNetParts finished"
echo "   ok:   $ok"
echo "   fail: $fail"
if [[ $fail -gt 0 ]]; then
    echo "   failures:"
    for f in "${failures[@]}"; do echo "     - $f"; done
fi
echo "============================================================"
