# Developer notes (local)

This repo lives on an external **exFAT** drive (`/media/jch903/fidelio1/...`). exFAT does **not** support the symlinks Python virtualenvs often use, so **do not** create `.venv/` inside the repo.

## Recommended setup

Use Python 3.13 (available on this machine) and keep the venv under your home directory:

```bash
python3.13 -m venv "$HOME/.venvs/PortalRecruit"
source "$HOME/.venvs/PortalRecruit/bin/activate"
python -m pip install -U pip
pip install -r requirements.txt
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

