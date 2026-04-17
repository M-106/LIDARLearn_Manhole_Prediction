#!/bin/bash
# ============================================================
# Train all models on HELIALS dataset
# Includes: supervised, SSL (full FT), and PEFT strategies
# Usage: bash scripts/train_helials.sh [--seed 42] [--max_epoch 250]
# ============================================================

SEED=${SEED:-42}
MAX_EPOCH=${MAX_EPOCH:-250}
AUGMENTATION=${AUGMENTATION:-none}

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2 ;;
        --max_epoch) MAX_EPOCH="$2"; shift 2 ;;
        --augmentation) AUGMENTATION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " HELIALS - All Models"
echo " Seed: $SEED | Epochs: $MAX_EPOCH | Aug: $AUGMENTATION"
echo "============================================================"

COMMON="--mode finetune --seed $SEED --max_epoch $MAX_EPOCH --augmentation $AUGMENTATION"

# ────────────────────────────────────────────────
#  SUPERVISED MODELS
# ────────────────────────────────────────────────

echo ""; echo "=== Supervised Models ==="

echo "[1/30] PointNet"
python main.py --config cfgs/classification/PointNet/HELIALS/helials.yaml $COMMON --exp_name pointnet_helials

echo "[2/30] PointNet++ (SSG)"
python main.py --config cfgs/classification/PointNet2/HELIALS/helials_ssg.yaml $COMMON --exp_name pointnet2_ssg_helials

echo "[3/30] PointNet++ (MSG)"
python main.py --config cfgs/classification/PointNet2/HELIALS/helials_msg.yaml $COMMON --exp_name pointnet2_msg_helials

echo "[4/30] DGCNN"
python main.py --config cfgs/classification/DGCNN/HELIALS/helials.yaml $COMMON --exp_name dgcnn_helials

echo "[5/30] PCT"
python main.py --config cfgs/classification/PCT/HELIALS/helials.yaml $COMMON --exp_name pct_helials

echo "[6/30] PointMLP"
python main.py --config cfgs/classification/PointMLP/HELIALS/helials.yaml $COMMON --exp_name pointmlp_helials

echo "[7/30] CurveNet"
python main.py --config cfgs/classification/CurveNet/HELIALS/helials.yaml $COMMON --exp_name curvenet_helials

echo "[8/30] DeepGCN"
python main.py --config cfgs/classification/DeepGCN/HELIALS/helials.yaml $COMMON --exp_name deepgcn_helials

echo "[9/30] DELA"
python main.py --config cfgs/classification/DELA/HELIALS/helials.yaml $COMMON --exp_name dela_helials

echo "[10/30] RSCNN"
python main.py --config cfgs/classification/RSCNN/HELIALS/helials.yaml $COMMON --exp_name rscnn_helials

echo "[11/30] PointConv"
python main.py --config cfgs/classification/PointConv/HELIALS/helials.yaml $COMMON --exp_name pointconv_helials

echo "[12/30] PointWeb"
python main.py --config cfgs/classification/PointWeb/HELIALS/helials.yaml $COMMON --exp_name pointweb_helials

echo "[13/30] SO-Net"
python main.py --config cfgs/classification/SONet/HELIALS/helials.yaml $COMMON --exp_name sonet_helials

echo "[14/30] RepSurf"
python main.py --config cfgs/classification/RepSurf/HELIALS/helials.yaml $COMMON --exp_name repsurf_helials

echo "[15/30] PointCNN"
python main.py --config cfgs/classification/PointCNN/HELIALS/helials.yaml $COMMON --exp_name pointcnn_helials

echo "[16/30] PointSCNet"
python main.py --config cfgs/classification/PointSCNet/HELIALS/helials.yaml $COMMON --exp_name pointscnet_helials

echo "[17/30] GDANet"
python main.py --config cfgs/classification/GDAN/HELIALS/helials.yaml $COMMON --exp_name gdan_helials

echo "[19/30] PPFNet"
python main.py --config cfgs/classification/PPFNet/HELIALS/helials.yaml $COMMON --exp_name ppfnet_helials

echo "[20/30] PVT"
python main.py --config cfgs/classification/PVT/HELIALS/helials.yaml $COMMON --exp_name pvt_helials

echo "[21/30] PointTransformer"
python main.py --config cfgs/classification/PointTransformer/HELIALS/helials.yaml $COMMON --exp_name pointtransformer_helials

echo "[22/30] PointTransformerV2"
python main.py --config cfgs/classification/PointTransformerV2/HELIALS/helials.yaml $COMMON --exp_name pointtransformerv2_helials

echo "[23/30] PointTransformerV3"
python main.py --config cfgs/classification/PointTransformerV3/HELIALS/helials.yaml $COMMON --exp_name pointtransformerv3_helials

echo "[24/30] P2P"
python main.py --config cfgs/classification/P2P/HELIALS/helials.yaml $COMMON --exp_name p2p_helials

echo "[25/30] PointTNT"
python main.py --config cfgs/classification/PointTNT/HELIALS/helials.yaml $COMMON --exp_name pointtnt_helials

echo "[26/30] GlobalTransformer"
python main.py --config cfgs/classification/GlobalTransformer/HELIALS/helials.yaml $COMMON --exp_name globaltransformer_helials

echo "[27/30] PointKAN"
python main.py --config cfgs/classification/PointKAN/HELIALS/helials.yaml $COMMON --exp_name pointkan_helials

echo "[28/30] MSDGCNN"
python main.py --config cfgs/classification/MSDGCNN/HELIALS/helials.yaml $COMMON --exp_name msdgcnn_helials

echo "[29/30] MSDGCNN2"
python main.py --config cfgs/classification/MSDGCNN2/HELIALS/helials.yaml $COMMON --exp_name msdgcnn2_helials

echo "[30/30] KAN-DGCNN"
python main.py --config cfgs/classification/KANDGCNN/HELIALS/helials.yaml $COMMON --exp_name kandgcnn_helials

# ────────────────────────────────────────────────
#  SSL MODELS — Full Fine-tuning
# ────────────────────────────────────────────────

echo ""; echo "=== SSL Models (Full Fine-tuning) ==="

echo "[SSL 1/7] PointMAE"
python main.py --config cfgs/classification/PointMAE/HELIALS/helials.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_helials

echo "[SSL 2/7] PointBERT"
python main.py --config cfgs/classification/PointBERT/HELIALS/helials.yaml $COMMON --ckpts pretrained/pretrained_bert.pth --exp_name pointbert_helials

echo "[SSL 3/7] PointGPT"
python main.py --config cfgs/classification/PointGPT/HELIALS/helials.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_helials

echo "[SSL 4/7] ACT"
python main.py --config cfgs/classification/ACT/HELIALS/helials.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_helials

echo "[SSL 5/7] ReCon"
python main.py --config cfgs/classification/RECON/HELIALS/helials.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_helials

echo "[SSL 6/7] PCP-MAE"
python main.py --config cfgs/classification/PCPMAE/HELIALS/helials.yaml $COMMON --ckpts pretrained/pretrained_pcp.pth --exp_name pcpmae_helials

echo "[SSL 7/7] Point-M2AE"
python main.py --config cfgs/classification/PointM2AE/HELIALS/helials.yaml $COMMON --ckpts pretrained/pretrained_m2ae.pth --exp_name pointm2ae_helials

# ────────────────────────────────────────────────
#  PEFT STRATEGIES (PointMAE, PointGPT, ACT, ReCon)
# ────────────────────────────────────────────────

echo ""; echo "=== PEFT: PointMAE ==="

python main.py --config cfgs/classification/PointMAE/HELIALS/helials_idpt.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_idpt_helials
python main.py --config cfgs/classification/PointMAE/HELIALS/helials_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_vpt_deep_helials
python main.py --config cfgs/classification/PointMAE/HELIALS/helials_dapt.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_dapt_helials
python main.py --config cfgs/classification/PointMAE/HELIALS/helials_ppt.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_ppt_helials
python main.py --config cfgs/classification/PointMAE/HELIALS/helials_gst.yaml $COMMON --ckpts pretrained/pretrained_mae.pth --exp_name pointmae_gst_helials

echo ""; echo "=== PEFT: PointGPT ==="

python main.py --config cfgs/classification/PointGPT/HELIALS/helials_idpt.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_idpt_helials
python main.py --config cfgs/classification/PointGPT/HELIALS/helials_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_vpt_deep_helials
python main.py --config cfgs/classification/PointGPT/HELIALS/helials_dapt.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_dapt_helials
python main.py --config cfgs/classification/PointGPT/HELIALS/helials_ppt.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_ppt_helials
python main.py --config cfgs/classification/PointGPT/HELIALS/helials_gst.yaml $COMMON --ckpts pretrained/pretrained_gpt.pth --exp_name pointgpt_gst_helials

echo ""; echo "=== PEFT: ACT ==="

python main.py --config cfgs/classification/ACT/HELIALS/helials_idpt.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_idpt_helials
python main.py --config cfgs/classification/ACT/HELIALS/helials_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_vpt_deep_helials
python main.py --config cfgs/classification/ACT/HELIALS/helials_dapt.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_dapt_helials
python main.py --config cfgs/classification/ACT/HELIALS/helials_ppt.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_ppt_helials
python main.py --config cfgs/classification/ACT/HELIALS/helials_gst.yaml $COMMON --ckpts pretrained/pretrained_act.pth --exp_name act_gst_helials

echo ""; echo "=== PEFT: ReCon ==="

python main.py --config cfgs/classification/RECON/HELIALS/helials_idpt.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_idpt_helials
python main.py --config cfgs/classification/RECON/HELIALS/helials_vpt_deep.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_vpt_deep_helials
python main.py --config cfgs/classification/RECON/HELIALS/helials_dapt.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_dapt_helials
python main.py --config cfgs/classification/RECON/HELIALS/helials_ppt.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_ppt_helials
python main.py --config cfgs/classification/RECON/HELIALS/helials_gst.yaml $COMMON --ckpts pretrained/pretrained_recon.pth --exp_name recon_gst_helials

echo ""
echo "============================================================"
echo " HELIALS Training Complete"
echo "============================================================"
