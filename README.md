# PortalRecruit 🏀

**PortalRecruit is an AI scouting engine for college basketball coaches.**
It turns coach‑speak into precise search, surfaces hidden impact traits, and delivers instant, actionable player intel—without spreadsheets.

---

## ✨ What it does
- **Natural‑language search** for prospects (“guard who can defend late clock”).
- **Trait‑driven rankings** (dog, menace, rim pressure, gravity, etc.).
- **Player profiles** with stats snapshots, scouting summaries, and film context.
- **ACC 2021–22 data pipeline** (PDF parsing + DB ingestion).
- **Social Media Scout (beta)**: queue‑driven report generation with LLM analysis.

---

## 🧭 Quick Start

```bash
# 1) Create venv
python3 -m venv ~/.venv_310
source ~/.venv_310/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run app
streamlit run src/dashboard/Home.py
```

---

## 🔍 Data & Search

PortalRecruit combines:
- **Vector search** (Chroma)
- **Reranking** (cross‑encoder)
- **Trait + stats signals**
- **Coach‑speak intent parsing**

Search results are ranked with blended scoring for precision, speed, and interpretability.

---

## 📊 Player Profiles
Profiles surface:
- Position / school / height / weight
- Stats snapshot (boxscore + per‑game)
- Trait strengths/weaknesses
- Film context (tagged clips)
- LLM scouting summary
- Social media report (when available)

---

## 🧪 Social Media Scout (Beta)
Queue‑driven pipeline:
1) Search (Serper.dev)
2) Verify (LLM)
3) Scrape (Instagram via Instaloader)
4) Analyze (LLM)

Run worker:
```bash
export SERPER_API_KEY="..."
export OPENAI_API_KEY="..."
source ~/.venv_310/bin/activate
python scripts/social_scout_worker.py
```

---

## 🗂️ Project Structure
```
PortalRecruit/
├── src/
│   ├── dashboard/        # Streamlit UI
│   ├── ingestion/        # DB + pipelines
│   ├── search/           # semantic search + rerank
│   └── ml/               # models & training
├── scripts/              # backfills, ingests, workers
├── data/                 # skout.db, vector_db
└── www/                  # branding + CSS
```

---

## 🔐 Environment Variables
```
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o
SERPER_API_KEY=...
```

---

## ✅ Notes
- Streamlit entry: `src/dashboard/Home.py`
- ACC focus: 2021–2022 (current data scope)
- DB: `data/skout.db`

---

## 🤝 Contributing
If you’re a coach, analyst, or engineer and want to help improve PortalRecruit, open a PR or message the team.

---

**PortalRecruit = Search > Recruit > Win**
