#!/bin/bash
# ============================================================
# Train all models on ShapeNet Parts (part segmentation)
# Usage: bash scripts/train_shapenetparts.sh [--seed 42] [--npoints 2048]
# ============================================================

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-50}
NPOINTS=${NPOINTS:-1024}

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

    "PointBERT:cfgs/segmentation/PointBERT/ShapeNetParts/pointbert_partseg.yaml:pretrained/pretrained_bert.pth"
   
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

   
)

for entry in "${PEFT_MODELS[@]}"; do
    IFS=':' read -r NAME CFG CKPT <<< "$entry"
    echo "[PEFT] $NAME"
    python main.py --config "$CFG" $COMMON --ckpts "$CKPT" --exp_name "${NAME,,}_partseg"
done

echo "============================================================"
echo " ShapeNet Parts Training Complete"
echo "============================================================"
