#!/bin/bash
# ============================================================
# Train all models on S3DIS (semantic segmentation)
# Usage: bash scripts/train_s3dis.sh [--seed 42] [--npoints 4096]
# ============================================================

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-1}
NPOINTS=${NPOINTS:-256}

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2 ;;
        --max_epoch) MAX_EPOCH="$2"; shift 2 ;;
        --npoints) NPOINTS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " S3DIS Training - All Models (Semantic Segmentation)"
echo " Seed: $SEED | Max Epoch: $MAX_EPOCH | N Points: $NPOINTS"
echo "============================================================"

COMMON="--mode seg --seed $SEED --max_epoch $MAX_EPOCH"

# ── Supervised Models ──

MODELS=(
    # PointCentricSeg wrapper (hook-based, all N points preserved)
    "PointNet:cfgs/segmentation/PointNet/S3DIS/pointnet_s3dis.yaml"
    "DGCNN:cfgs/segmentation/DGCNN/S3DIS/dgcnn_s3dis.yaml"
    "MSDGCNN:cfgs/segmentation/MSDGCNN/S3DIS/msdgcnn_s3dis.yaml"
    "MSDGCNN2:cfgs/segmentation/MSDGCNN2/S3DIS/msdgcnn2_s3dis.yaml"
    "GDAN:cfgs/segmentation/GDAN/S3DIS/gdan_s3dis.yaml"
    "KANDGCNN:cfgs/segmentation/KANDGCNN/S3DIS/kandgcnn_s3dis.yaml"
    "PointKAN:cfgs/segmentation/PointKAN/S3DIS/pointkan_s3dis.yaml"
    "DeepGCN:cfgs/segmentation/DeepGCN/S3DIS/deepgcn_s3dis.yaml"
    # Dedicated encoder-decoder
    "PointNet2:cfgs/segmentation/PointNet2/S3DIS/pointnet2_s3dis.yaml"
    "CurveNet:cfgs/segmentation/CurveNet/S3DIS/curvenet_s3dis.yaml"
    "DELA:cfgs/segmentation/DELA/S3DIS/dela_s3dis.yaml"
    "PCT:cfgs/segmentation/PCT/S3DIS/pct_s3dis.yaml"
    "PointSCNet:cfgs/segmentation/PointSCNet/S3DIS/pointscnet_s3dis.yaml"
    "RSCNN:cfgs/segmentation/RSCNN/S3DIS/rscnn_s3dis.yaml"
    "PointConv:cfgs/segmentation/PointConv/S3DIS/pointconv_s3dis.yaml"
    "PVT:cfgs/segmentation/PVT/S3DIS/pvt_s3dis.yaml"
    "PointWeb:cfgs/segmentation/PointWeb/S3DIS/pointweb_s3dis.yaml"
    "PointMLP:cfgs/segmentation/PointMLP/S3DIS/pointmlp_s3dis.yaml"
    "RandLANet:cfgs/segmentation/RandLANet/S3DIS/randlanet_s3dis.yaml"
    "RepSurf:cfgs/segmentation/RepSurf/S3DIS/repsurf_s3dis.yaml"
    "PointTransformer:cfgs/segmentation/PointTransformer/S3DIS/pointtransformer_s3dis.yaml"
    "PointTransformerV2:cfgs/segmentation/PointTransformerV2/S3DIS/pointtransformerv2_s3dis.yaml"
    "P2P:cfgs/segmentation/P2P/S3DIS/p2p_s3dis.yaml"
    "GlobalTransformer:cfgs/segmentation/GlobalTransformer/S3DIS/globaltransformer_s3dis.yaml"
    "PointTNT:cfgs/segmentation/PointTNT/S3DIS/pointtnt_s3dis.yaml"
    "PointTransformerV3:cfgs/segmentation/PointTransformerV3/S3DIS/pointtransformerv3_s3dis.yaml"
)

COUNT=1
TOTAL=${#MODELS[@]}
for entry in "${MODELS[@]}"; do
    NAME="${entry%%:*}"
    CFG="${entry##*:}"
    echo "[$COUNT/$TOTAL] $NAME"
    python main.py --config "$CFG" $COMMON --exp_name "${NAME,,}_s3dis"
    COUNT=$((COUNT + 1))
done

# ── SSL Models (Full Fine-tuning) ──

SSL_MODELS=(
    "PointMAE:cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis.yaml:pretrained/pretrained_mae.pth"
    "PointBERT:cfgs/segmentation/PointBERT/S3DIS/pointbert_s3dis.yaml:pretrained/pretrained_bert.pth"
    "PointGPT:cfgs/segmentation/PointGPT/S3DIS/pointgpt_s3dis.yaml:pretrained/pretrained_gpt.pth"
    "ACT:cfgs/segmentation/ACT/S3DIS/act_s3dis.yaml:pretrained/pretrained_act.pth"
    "ReCon:cfgs/segmentation/RECON/S3DIS/recon_s3dis.yaml:pretrained/pretrained_recon.pth"
    "PCPMAE:cfgs/segmentation/PCPMAE/S3DIS/pcpmae_s3dis.yaml:pretrained/pretrained_pcp.pth"
    "PointM2AE:cfgs/segmentation/PointM2AE/S3DIS/pointm2ae_s3dis.yaml:pretrained/pretrained_m2ae.pth"
)

for entry in "${SSL_MODELS[@]}"; do
    IFS=':' read -r NAME CFG CKPT <<< "$entry"
    echo "[SSL] $NAME"
    python main.py --config "$CFG" $COMMON --ckpts "$CKPT" --exp_name "${NAME,,}_s3dis"
done

# ── PEFT Models (PointMAE / ACT / ReCon backbones) ──

PEFT_MODELS=(
    # PPT
    "PointMAE-PPT:cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis_ppt.yaml:pretrained/pretrained_mae.pth"
    "ACT-PPT:cfgs/segmentation/ACT/S3DIS/act_s3dis_ppt.yaml:pretrained/pretrained_act.pth"
    "ReCon-PPT:cfgs/segmentation/RECON/S3DIS/recon_s3dis_ppt.yaml:pretrained/pretrained_recon.pth"
    # DAPT
    "PointMAE-DAPT:cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis_dapt.yaml:pretrained/pretrained_mae.pth"
    "ACT-DAPT:cfgs/segmentation/ACT/S3DIS/act_s3dis_dapt.yaml:pretrained/pretrained_act.pth"
    "ReCon-DAPT:cfgs/segmentation/RECON/S3DIS/recon_s3dis_dapt.yaml:pretrained/pretrained_recon.pth"
    # IDPT
    "PointMAE-IDPT:cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis_idpt.yaml:pretrained/pretrained_mae.pth"
    "ACT-IDPT:cfgs/segmentation/ACT/S3DIS/act_s3dis_idpt.yaml:pretrained/pretrained_act.pth"
    "ReCon-IDPT:cfgs/segmentation/RECON/S3DIS/recon_s3dis_idpt.yaml:pretrained/pretrained_recon.pth"
    # PointGST
    "PointMAE-GST:cfgs/segmentation/PointMAE/S3DIS/pointmae_s3dis_gst.yaml:pretrained/pretrained_mae.pth"
    "ACT-GST:cfgs/segmentation/ACT/S3DIS/act_s3dis_gst.yaml:pretrained/pretrained_act.pth"
    "ReCon-GST:cfgs/segmentation/RECON/S3DIS/recon_s3dis_gst.yaml:pretrained/pretrained_recon.pth"
)

for entry in "${PEFT_MODELS[@]}"; do
    IFS=':' read -r NAME CFG CKPT <<< "$entry"
    echo "[PEFT] $NAME"
    python main.py --config "$CFG" $COMMON --ckpts "$CKPT" --exp_name "${NAME,,}_s3dis"
done

echo "============================================================"
echo " S3DIS Training Complete"
echo "============================================================"
