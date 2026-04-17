#!/bin/bash
# ============================================================
# Train all models on ShapeNet Parts (part segmentation)
# Usage: bash scripts/train_shapenetparts.sh [--seed 42] [--npoints 2048]
# ============================================================

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-1}
NPOINTS=${NPOINTS:-512}

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2 ;;
        --max_epoch) MAX_EPOCH="$2"; shift 2 ;;
        --npoints) NPOINTS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " ShapeNet Parts - All Models (Part Segmentation)"
echo " Seed: $SEED | Max Epoch: $MAX_EPOCH | N Points: $NPOINTS"
echo "============================================================"

COMMON="--mode seg --seed $SEED --max_epoch $MAX_EPOCH"

# ── Supervised Models ──

MODELS=(
    # PointCentricSeg wrapper (hook-based, all N points preserved)
    "PointNet:cfgs/segmentation/PointNet/ShapeNetParts/pointnet_partseg.yaml"
    "DGCNN:cfgs/segmentation/DGCNN/ShapeNetParts/dgcnn_partseg.yaml"
    "MSDGCNN:cfgs/segmentation/MSDGCNN/ShapeNetParts/msdgcnn_partseg.yaml"
    "MSDGCNN2:cfgs/segmentation/MSDGCNN2/ShapeNetParts/msdgcnn2_partseg.yaml"
    "GDAN:cfgs/segmentation/GDAN/ShapeNetParts/gdan_partseg.yaml"
    "KANDGCNN:cfgs/segmentation/KANDGCNN/ShapeNetParts/kandgcnn_partseg.yaml"
    "PointKAN:cfgs/segmentation/PointKAN/ShapeNetParts/pointkan_partseg.yaml"
    "DeepGCN:cfgs/segmentation/DeepGCN/ShapeNetParts/deepgcn_partseg.yaml"
    # Dedicated encoder-decoder
    "PointNet2:cfgs/segmentation/PointNet2/ShapeNetParts/pointnet2_partseg.yaml"
    "CurveNet:cfgs/segmentation/CurveNet/ShapeNetParts/curvenet_partseg.yaml"
    "DELA:cfgs/segmentation/DELA/ShapeNetParts/dela_partseg.yaml"
    "PCT:cfgs/segmentation/PCT/ShapeNetParts/pct_partseg.yaml"
    "PointSCNet:cfgs/segmentation/PointSCNet/ShapeNetParts/pointscnet_partseg.yaml"
    "RSCNN:cfgs/segmentation/RSCNN/ShapeNetParts/rscnn_partseg.yaml"
    "PointConv:cfgs/segmentation/PointConv/ShapeNetParts/pointconv_partseg.yaml"
    "PVT:cfgs/segmentation/PVT/ShapeNetParts/pvt_partseg.yaml"
    "PointWeb:cfgs/segmentation/PointWeb/ShapeNetParts/pointweb_partseg.yaml"
    "PointMLP:cfgs/segmentation/PointMLP/ShapeNetParts/pointmlp_partseg.yaml"
    "RandLANet:cfgs/segmentation/RandLANet/ShapeNetParts/randlanet_partseg.yaml"
    "RepSurf:cfgs/segmentation/RepSurf/ShapeNetParts/repsurf_partseg.yaml"
    "PointTransformer:cfgs/segmentation/PointTransformer/ShapeNetParts/pointtransformer_partseg.yaml"
    "PointTransformerV2:cfgs/segmentation/PointTransformerV2/ShapeNetParts/pointtransformerv2_partseg.yaml"
    "P2P:cfgs/segmentation/P2P/ShapeNetParts/p2p_partseg.yaml"
    "GlobalTransformer:cfgs/segmentation/GlobalTransformer/ShapeNetParts/globaltransformer_partseg.yaml"
    "PointTNT:cfgs/segmentation/PointTNT/ShapeNetParts/pointtnt_partseg.yaml"
    "PointTransformerV3:cfgs/segmentation/PointTransformerV3/ShapeNetParts/pointtransformerv3_partseg.yaml"
)

COUNT=1
TOTAL=${#MODELS[@]}
for entry in "${MODELS[@]}"; do
    NAME="${entry%%:*}"
    CFG="${entry##*:}"
    echo "[$COUNT/$TOTAL] $NAME"
    python main.py --config "$CFG" $COMMON --exp_name "${NAME,,}_partseg"
    COUNT=$((COUNT + 1))
done

# ── SSL Models (Full Fine-tuning) ──

SSL_MODELS=(
    "PointMAE:cfgs/segmentation/PointMAE/ShapeNetParts/pointmae_partseg.yaml:pretrained/pretrained_mae.pth"
    "PointBERT:cfgs/segmentation/PointBERT/ShapeNetParts/pointbert_partseg.yaml:pretrained/pretrained_bert.pth"
    "PointGPT:cfgs/segmentation/PointGPT/ShapeNetParts/pointgpt_partseg.yaml:pretrained/pretrained_gpt.pth"
    "ACT:cfgs/segmentation/ACT/ShapeNetParts/act_partseg.yaml:pretrained/pretrained_act.pth"
    "ReCon:cfgs/segmentation/RECON/ShapeNetParts/recon_partseg.yaml:pretrained/pretrained_recon.pth"
    "PCPMAE:cfgs/segmentation/PCPMAE/ShapeNetParts/pcpmae_partseg.yaml:pretrained/pretrained_pcp.pth"
    "PointM2AE:cfgs/segmentation/PointM2AE/ShapeNetParts/pointm2ae_partseg.yaml:pretrained/pretrained_m2ae.pth"
)

for entry in "${SSL_MODELS[@]}"; do
    IFS=':' read -r NAME CFG CKPT <<< "$entry"
    echo "[SSL] $NAME"
    python main.py --config "$CFG" $COMMON --ckpts "$CKPT" --exp_name "${NAME,,}_partseg"
done

# ── PEFT Models (PointMAE / ACT / ReCon backbones) ──

PEFT_MODELS=(
    # PPT
    "PointMAE-PPT:cfgs/segmentation/PointMAE/ShapeNetParts/pointmae_partseg_ppt.yaml:pretrained/pretrained_mae.pth"
    "ACT-PPT:cfgs/segmentation/ACT/ShapeNetParts/act_partseg_ppt.yaml:pretrained/pretrained_act.pth"
    "ReCon-PPT:cfgs/segmentation/RECON/ShapeNetParts/recon_partseg_ppt.yaml:pretrained/pretrained_recon.pth"
    # DAPT
    "PointMAE-DAPT:cfgs/segmentation/PointMAE/ShapeNetParts/pointmae_partseg_dapt.yaml:pretrained/pretrained_mae.pth"
    "ACT-DAPT:cfgs/segmentation/ACT/ShapeNetParts/act_partseg_dapt.yaml:pretrained/pretrained_act.pth"
    "ReCon-DAPT:cfgs/segmentation/RECON/ShapeNetParts/recon_partseg_dapt.yaml:pretrained/pretrained_recon.pth"
    # IDPT
    "PointMAE-IDPT:cfgs/segmentation/PointMAE/ShapeNetParts/pointmae_partseg_idpt.yaml:pretrained/pretrained_mae.pth"
    "ACT-IDPT:cfgs/segmentation/ACT/ShapeNetParts/act_partseg_idpt.yaml:pretrained/pretrained_act.pth"
    "ReCon-IDPT:cfgs/segmentation/RECON/ShapeNetParts/recon_partseg_idpt.yaml:pretrained/pretrained_recon.pth"
    # PointGST
    "PointMAE-GST:cfgs/segmentation/PointMAE/ShapeNetParts/pointmae_partseg_gst.yaml:pretrained/pretrained_mae.pth"
    "ACT-GST:cfgs/segmentation/ACT/ShapeNetParts/act_partseg_gst.yaml:pretrained/pretrained_act.pth"
    "ReCon-GST:cfgs/segmentation/RECON/ShapeNetParts/recon_partseg_gst.yaml:pretrained/pretrained_recon.pth"
)

for entry in "${PEFT_MODELS[@]}"; do
    IFS=':' read -r NAME CFG CKPT <<< "$entry"
    echo "[PEFT] $NAME"
    python main.py --config "$CFG" $COMMON --ckpts "$CKPT" --exp_name "${NAME,,}_partseg"
done

echo "============================================================"
echo " ShapeNet Parts Training Complete"
echo "============================================================"
