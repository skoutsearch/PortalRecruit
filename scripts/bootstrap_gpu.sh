#!/usr/bin/env bash
set -euo pipefail

# GPU bootstrap: installs CUDA-enabled PyTorch matching the host's driver-supported CUDA.
# Host (this machine): NVIDIA driver 555.xx, CUDA 12.5 (see `nvidia-smi`).
#
# Notes:
# - PyTorch wheels are built for specific CUDA toolkits (e.g. cu121, cu124, cu126).
# - We default to cu124 for CUDA 12.5-class drivers (driver >= 550) since it is broadly available.
# - If you want a different wheel set, set TORCH_INDEX_URL explicitly.

VENV_DIR="${VENV_DIR:-$HOME/.venvs/PortalRecruit}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

# Default: CUDA 12.4 wheels, compatible with CUDA 12.5 driver
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "[PortalRecruit] Using venv: $VENV_DIR"

echo "[PortalRecruit] Torch index: $TORCH_INDEX_URL"

echo "[PortalRecruit] Creating/using Python 3.13 venv"
if [[ -d "$VENV_DIR" && "$FORCE_RECREATE" != "1" ]]; then
  echo "[PortalRecruit] Venv already exists. Set FORCE_RECREATE=1 to recreate it."
else
  rm -rf "$VENV_DIR"
  python3.13 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -U pip

# Install CUDA torch first
"$VENV_DIR/bin/pip" install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio

# Then project deps
"$VENV_DIR/bin/pip" install -r requirements.txt

# Quick check
"$VENV_DIR/bin/python" - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY

echo ""
echo "Done. Activate with:"
echo "  source \"$VENV_DIR/bin/activate\""
