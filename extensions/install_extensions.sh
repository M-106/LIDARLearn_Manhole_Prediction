#!/bin/bash
# ============================================================
# Install all CUDA C++ extensions for LIDARLearn
#
# Usage:
#   bash scripts/install_extensions.sh          # install all
#   bash scripts/install_extensions.sh --clean  # clean + rebuild
#
# Requirements:
#   - CUDA toolkit matching your PyTorch installation
#   - ninja (pip install ninja) for faster builds
#   - GCC compatible with your CUDA version
#
# GPU architecture is auto-detected. To override (e.g. cross-compile):
#   export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="$(cd "$SCRIPT_DIR/../extensions" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CLEAN=false
for arg in "$@"; do
    case $arg in
        --clean) CLEAN=true ;;
    esac
done

# ── Clean ──
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    find "$EXT_DIR" -name "build" -type d -exec rm -rf {} + 2>/dev/null
    find "$EXT_DIR" -name "dist" -type d -exec rm -rf {} + 2>/dev/null
    find "$EXT_DIR" -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null
    find "$EXT_DIR" -name "*.so" -delete 2>/dev/null
    find "$EXT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

    # Remove stale pip installations pointing to old extension paths
    echo -e "${YELLOW}Removing stale pip installations...${NC}"
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
    if [ -n "$SITE_PACKAGES" ]; then
        pip uninstall -y pointnet2_ops chamfer chamfer_dist dela_cutils \
            pointops point_transformer_ops ptv_modules clip emd_ext \
            pointnet2_cuda index_max 2>/dev/null || true
        # Clean orphaned egg-links pointing to LIDARLearn
        find "$SITE_PACKAGES" -maxdepth 1 -name "*.egg-link" \
            -exec grep -l "LIDARLearn/extensions" {} \; -delete 2>/dev/null || true
        # Clean orphaned directories
        rm -rf "$SITE_PACKAGES/pointnet2_ops" 2>/dev/null || true
    fi

    echo -e "${GREEN}Done.${NC}"
    echo ""
fi

# ── Prerequisites ──
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null || {
    echo -e "${RED}Error: PyTorch with CUDA support is required.${NC}"
    exit 1
}

CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
GPU_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "?")

echo "PyTorch $TORCH_VERSION | CUDA $CUDA_VERSION | GPU: $GPU_NAME (sm_$GPU_ARCH)"
echo ""

# ── Install function ──
PASS=0
FAIL=0

install_ext() {
    local name="$1"
    local dir="$2"

    echo -ne "${YELLOW}[$name]${NC} Building... "
    local log
    log=$(cd "$dir" && pip install -e . --no-build-isolation 2>&1)
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo -e "${GREEN}OK${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAILED${NC}"
        echo "$log" | tail -5
        FAIL=$((FAIL + 1))
    fi
}

# ── Core extensions ──
echo "============================================================"
echo " Core Extensions"
echo "============================================================"

install_ext "pointnet2_ops"       "$EXT_DIR/pointnet2_ops"
install_ext "chamfer_dist"        "$EXT_DIR/chamfer_dist"

# ── Model-specific extensions ──
echo ""
echo "============================================================"
echo " Model-Specific Extensions"
echo "============================================================"

install_ext "dela_cutils"         "$EXT_DIR/dela_cutils"
install_ext "pointops"            "$EXT_DIR/pointops"
install_ext "ptv_modules"         "$EXT_DIR/ptv_modules"
install_ext "clip"                "$EXT_DIR/clip"
install_ext "index_max"           "$EXT_DIR/index_max_ext"

echo ""
echo "============================================================"
echo -e " ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC} (out of 7 extensions)"
echo "============================================================"
