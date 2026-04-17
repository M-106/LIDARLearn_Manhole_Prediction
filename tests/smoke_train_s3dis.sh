#!/bin/bash
# ============================================================
# SMOKE TEST: S3DIS semantic segmentation
#   * 1 epoch per model
#   * --data_fraction 0.05 (5% of train + val blocks)
#
# Validates every config under cfgs/segmentation/*/S3DIS/
# loads + completes one epoch. Runs in ~5-15 minutes on a
# single GPU depending on batch size. Failures are tallied.
# ============================================================

set -u

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-1}
DATA_FRACTION=${DATA_FRACTION:-0.05}

COMMON="--mode seg --seed $SEED --max_epoch $MAX_EPOCH --data_fraction $DATA_FRACTION"

declare -a JOBS=(
    # Supervised
    "pointnet|cfgs/segmentation/PointNet/S3DIS/pointnet_s3dis.yaml|"
    "dgcnn|cfgs/segmentation/DGCNN/S3DIS/dgcnn_s3dis.yaml|"
    "msdgcnn|cfgs/segmentation/MSDGCNN/S3DIS/msdgcnn_s3dis.yaml|"
    "msdgcnn2|cfgs/segmentation/MSDGCNN2/S3DIS/msdgcnn2_s3dis.yaml|"
    "gdan|cfgs/segmentation/GDAN/S3DIS/gdan_s3dis.yaml|"
    "kandgcnn|cfgs/segmentation/KANDGCNN/S3DIS/kandgcnn_s3dis.yaml|"
    "pointkan|cfgs/segmentation/PointKAN/S3DIS/pointkan_s3dis.yaml|"
    "deepgcn|cfgs/segmentation/DeepGCN/S3DIS/deepgcn_s3dis.yaml|"
    "pointnet2|cfgs/segmentation/PointNet2/S3DIS/pointnet2_s3dis.yaml|"
    "curvenet|cfgs/segmentation/CurveNet/S3DIS/curvenet_s3dis.yaml|"
    "dela|cfgs/segmentation/DELA/S3DIS/dela_s3dis.yaml|"
    "pct|cfgs/segmentation/PCT/S3DIS/pct_s3dis.yaml|"
    "pointscnet|cfgs/segmentation/PointSCNet/S3DIS/pointscnet_s3dis.yaml|"
    "rscnn|cfgs/segmentation/RSCNN/S3DIS/rscnn_s3dis.yaml|"
    "pointconv|cfgs/segmentation/PointConv/S3DIS/pointconv_s3dis.yaml|"
    "pvt|cfgs/segmentation/PVT/S3DIS/pvt_s3dis.yaml|"
    "pointweb|cfgs/segmentation/PointWeb/S3DIS/pointweb_s3dis.yaml|"
    "pointmlp|cfgs/segmentation/PointMLP/S3DIS/pointmlp_s3dis.yaml|"
    "randlanet|cfgs/segmentation/RandLANet/S3DIS/randlanet_s3dis.yaml|"
    "repsurf|cfgs/segmentation/RepSurf/S3DIS/repsurf_s3dis.yaml|"
    "pointtransformer|cfgs/segmentation/PointTransformer/S3DIS/pointtransformer_s3dis.yaml|"
    "pointtransformerv2|cfgs/segmentation/PointTransformerV2/S3DIS/pointtransformerv2_s3dis.yaml|"
    "pointtransformerv3|cfgs/segmentation/PointTransformerV3/S3DIS/pointtransformerv3_s3dis.yaml|"
    "p2p|cfgs/segmentation/P2P/S3DIS/p2p_s3dis.yaml|"
    "globaltransformer|cfgs/segmentation/GlobalTransformer/S3DIS/globaltransformer_s3dis.yaml|"
    "pointtnt|cfgs/segmentation/PointTNT/S3DIS/pointtnt_s3dis.yaml|"

    # SSL
    "pointmae|cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis.yaml|pretrained/pretrained_mae.pth"
    "pointbert|cfgs/segmentation/PointBERT/S3DIS/pointbert_s3dis.yaml|pretrained/pretrained_bert.pth"
    "pointgpt|cfgs/segmentation/PointGPT/S3DIS/pointgpt_s3dis.yaml|pretrained/pretrained_gpt.pth"
    "act|cfgs/segmentation/ACT/S3DIS/act_s3dis.yaml|pretrained/pretrained_act.pth"
    "recon|cfgs/segmentation/RECON/S3DIS/recon_s3dis.yaml|pretrained/pretrained_recon.pth"
    "pcpmae|cfgs/segmentation/PCPMAE/S3DIS/pcpmae_s3dis.yaml|pretrained/pretrained_pcp.pth"
    "pointm2ae|cfgs/segmentation/PointM2AE/S3DIS/pointm2ae_s3dis.yaml|pretrained/pretrained_m2ae.pth"

    # PEFT
    "pointmae_ppt|cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis_ppt.yaml|pretrained/pretrained_mae.pth"
    "act_ppt|cfgs/segmentation/ACT/S3DIS/act_s3dis_ppt.yaml|pretrained/pretrained_act.pth"
    "recon_ppt|cfgs/segmentation/RECON/S3DIS/recon_s3dis_ppt.yaml|pretrained/pretrained_recon.pth"
    "pointmae_dapt|cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis_dapt.yaml|pretrained/pretrained_mae.pth"
    "act_dapt|cfgs/segmentation/ACT/S3DIS/act_s3dis_dapt.yaml|pretrained/pretrained_act.pth"
    "recon_dapt|cfgs/segmentation/RECON/S3DIS/recon_s3dis_dapt.yaml|pretrained/pretrained_recon.pth"
    "pointmae_idpt|cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis_idpt.yaml|pretrained/pretrained_mae.pth"
    "act_idpt|cfgs/segmentation/ACT/S3DIS/act_s3dis_idpt.yaml|pretrained/pretrained_act.pth"
    "recon_idpt|cfgs/segmentation/RECON/S3DIS/recon_s3dis_idpt.yaml|pretrained/pretrained_recon.pth"
    "pointmae_gst|cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis_gst.yaml|pretrained/pretrained_mae.pth"
    "act_gst|cfgs/segmentation/ACT/S3DIS/act_s3dis_gst.yaml|pretrained/pretrained_act.pth"
    "recon_gst|cfgs/segmentation/RECON/S3DIS/recon_s3dis_gst.yaml|pretrained/pretrained_recon.pth"
)

echo "============================================================"
echo " SMOKE: S3DIS — 1 epoch / model, data_fraction=$DATA_FRACTION"
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
    if python main.py --config "$cfg" $COMMON $ckpt_arg --exp_name "smoke_${label}_s3dis"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        failures+=("$label")
        echo "  [FAIL] $label — continuing"
    fi
done

echo ""
echo "============================================================"
echo " SMOKE: S3DIS finished"
echo "   ok:   $ok"
echo "   fail: $fail"
if [[ $fail -gt 0 ]]; then
    echo "   failures:"
    for f in "${failures[@]}"; do echo "     - $f"; done
fi
echo "============================================================"
