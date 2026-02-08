# PortalRecruit (Developer Guide) 🏀

PortalRecruit is an AI scouting engine for college basketball coaches. This README focuses on **developer setup, architecture, and workflows**.

---

## ✅ Prerequisites
- **Python 3.10**
- **Streamlit**
- **SQLite**
- **ffmpeg** (for media utilities)

---

## ⚡ Quick Start
```bash
# Create venv
python3 -m venv ~/.venv_310
source ~/.venv_310/bin/activate

# Install deps
pip install -r requirements.txt

# Run app
streamlit run src/dashboard/Home.py
```

---

## 🔐 Environment Variables
Create `.env` (repo root):
```
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o
SERPER_API_KEY=...
```

---

## 🧠 Search Architecture
Search pipeline combines:
1) **Vector retrieval** (Chroma)
2) **Cross‑encoder rerank**
3) **Trait/intent blending**
4) **Coach‑speak intent expansion**

Key files:
- `src/search/semantic.py`
- `src/search/coach_dictionary.py`

---

## 📊 Data Pipeline
Primary ingestion + backfill tools live under `scripts/`:
- `scripts/acc_stats_from_pdf.py`
- `scripts/ingest_acc_stats_from_pdf.py`
- `scripts/backfill_height_weight_from_synergy.py`
- `scripts/ingest_acc_roster_txt.py`
- `scripts/ingest_acc_hs_stats_txt.py`

DB schema in:
- `src/ingestion/db.py`

---

## 🗂 Project Structure
```
PortalRecruit/
├── src/
│   ├── dashboard/        # Streamlit UI
│   ├── ingestion/        # DB + ingestion pipelines
│   ├── search/           # semantic search + rerank
│   └── ml/               # models & training
├── scripts/              # backfills, ingests, workers
├── data/                 # skout.db, vector_db
└── www/                  # branding + CSS
```

---

## 🔄 Social Media Scout (Beta)
Queue‑driven pipeline:
1) **Search** (Serper.dev)
2) **Verify** (LLM)
3) **Scrape** (Instaloader)
4) **Analyze** (LLM)

Worker:
```bash
source ~/.venv_310/bin/activate
python scripts/social_scout_worker.py
```

---

## 🧪 Tests
```bash
pytest -q tests/test_semantic_search.py
```

---

## 🚀 Deployment Notes
- Streamlit entry: `src/dashboard/Home.py`
- Local DB: `data/skout.db`
- Vector DB: `data/vector_db/`

---

## 🛠 Common Tasks
**Rebuild vector DB**
```bash
python src/processing/generate_embeddings.py
```

**Backfill boxscore stats from plays**
```bash
python scripts/backfill_boxscore_from_plays.py
```

---

## ✅ License / Access
Private by default. Coordinate with the PortalRecruit team before sharing.

---

**PortalRecruit = Search > Recruit > Win**
