# Developer notes (local)

This repo lives on an external **exFAT** drive (`/media/jch903/fidelio1/...`). exFAT does **not** support the symlinks Python virtualenvs often use, so **do not** create `.venv/` inside the repo.

## Recommended setup

Use Python 3.13 (available on this machine) and keep the venv under your home directory:

```bash
python3.13 -m venv "$HOME/.venvs/PortalRecruit"
source "$HOME/.venvs/PortalRecruit/bin/activate"
python -m pip install -U pip
```

## Install dependencies (ML stack is core)

This project uses `sentence-transformers`, which depends on **PyTorch**.

### CPU (default)

If you install `torch` from PyPI on Linux it may pull CUDA wheels and can be **~1GB+**.
For most dev workflows here, **CPU torch** is sufficient.

Recommended:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt
```

### GPU / CUDA

Default on this host:
- NVIDIA driver: 555.58.02
- CUDA reported by nvidia-smi: 12.5

Recommended GPU install on this host (installs CUDA-enabled torch first, then repo deps):

```bash
./scripts/bootstrap_gpu.sh
```

You can override the torch wheel index if needed:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 FORCE_RECREATE=1 ./scripts/bootstrap_gpu.sh
```

(We default to cu124 wheels which are compatible with CUDA 12.5-class drivers.)

If you prefer to install torch manually, use PyTorch’s selector and then run requirements:

```bash
pip install -r requirements.txt
```

## One-command bootstrap

CPU:

```bash
./scripts/bootstrap_local.sh
```

GPU (expects you already installed CUDA-enabled torch):

```bash
./scripts/bootstrap_gpu.sh
```

## Doctor (sanity checks)

```bash
python scripts/doctor.py
```

## Run

```bash
source "$HOME/.venvs/PortalRecruit/bin/activate"
python run_portalrecruit.py
# or
streamlit run src/dashboard/Home.py
```

## Data + secrets

- `*.env` is ignored. Keep real secrets in `.env` locally.
- `data/` is ignored and can be large (video clips, chroma db, sqlite, etc.).

