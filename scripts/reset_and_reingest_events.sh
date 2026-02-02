#!/usr/bin/env bash
set -euo pipefail

# Reset + re-ingest plays with corrected period/clock mapping.
# Run from repo root: /media/jch903/fidelio/CLAUDOG/PortalRecruit

DB_PATH="data/skout.db"

if [[ ! -f "$DB_PATH" ]]; then
  echo "[!] $DB_PATH not found. If you intended a full reset, run: rm -f data/skout.db"
  exit 1
fi

if [[ -z "${SYNERGY_API_KEY:-}" ]]; then
  echo "[!] SYNERGY_API_KEY is not set in the environment."
  echo "    - If you have .env, run: set -a; source .env; set +a"
  echo "    - Or export it: export SYNERGY_API_KEY=..."
  exit 1
fi

echo "[1/4] Deleting plays table contents..."
sqlite3 "$DB_PATH" "DELETE FROM plays;"

echo "[2/4] Re-ingesting events via pipeline (schedule + events)..."
# Uses the current scripts and corrected pipeline.py
python -c "from src.ingestion.pipeline import PipelinePlan, run_pipeline; import os; 
plan=PipelinePlan(league_code='ncaamb', season_id=os.environ.get('SEASON_ID',''), team_ids=[], ingest_events=True);
assert plan.season_id, 'Set SEASON_ID env var to a valid Synergy season id (e.g., export SEASON_ID=...)';
print(run_pipeline(plan=plan, api_key=os.environ['SYNERGY_API_KEY']))"

echo "[3/5] Sanity check: counts..."
sqlite3 -header -column "$DB_PATH" "select count(*) as games from games;"
sqlite3 -header -column "$DB_PATH" "select count(*) as plays from plays;"

echo "[4/5] Sanity check: show sample plays with period + clock..."
sqlite3 -header -column "$DB_PATH" \
  "select game_id, period, clock_seconds, clock_display, substr(description,1,80) as description from plays where clock_seconds is not null order by random() limit 25;"

echo "[5/5] Done. Expected: period is small ints (1-4) and clock_display looks like MM:SS."
