#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/.venvs/PortalRecruit}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

echo "[PortalRecruit] Using venv: $VENV_DIR"

if [[ -d "$VENV_DIR" && "$FORCE_RECREATE" != "1" ]]; then
  echo "[PortalRecruit] Venv already exists. Set FORCE_RECREATE=1 to recreate it."
else
  rm -rf "$VENV_DIR"
  python3.13 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -U pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "Done. Activate with:"
echo "  source \"$VENV_DIR/bin/activate\""
