import os
import sys
import time
import requests
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result

# --- CONFIGURATION ---
CONFERENCE_FILTER = "Big 12"  # Exact string match for the conference you want
YEARS_TO_INGEST = [2024, 2025]  # The seasons you want (e.g., 2024-25, 2025-26)
SAFETY_DELAY_SECONDS = 2.0    # 2.0s = very safe. Do not lower this below 1.5s.

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ingestion_safety_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add project root to path to access existing pipeline logic
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

# Import your existing pipeline classes
# We assume these exist based on your previous logs
try:
    from src.ingestion.pipeline import PipelinePlan, run_pipeline
    from src.ingestion.capabilities import discover_capabilities
except ImportError:
    logging.error("Could not find 'src.ingestion'. Make sure you run this from the repo root.")
    sys.exit(1)

# --- THE SAFETY PATCH ---
# This overrides the standard requests.get to enforce a hard sleep.
# This guarantees that NO part of your codebase can speed through the API limits.

if not hasattr(requests, "_original_get"):
    requests._original_get = requests.get

def is_rate_limit_error(response):
    if response.status_code == 429:
        logging.warning("⚠️ 429 Rate Limit Hit. The safety delay will handle this, but backing off now.")
        return True
    return response.status_code >= 500

@retry(
    retry=retry_if_result(is_rate_limit_error),
    wait=wait_exponential(multiplier=2, min=4, max=60), # Aggressive backoff if we do hit a limit
    stop=stop_after_attempt(10)
)
def safe_get(*args, **kwargs):
    # 1. THE HARD PAUSE
    time.sleep(SAFETY_DELAY_SECONDS)
    
    # 2. Log the URL for audit trails (sanitized of keys if necessary)
    url = args[0] if len(args) > 0 else kwargs.get("url", "unknown_url")
    logging.info(f"⬇️ Fetching: {url} (Wait: {SAFETY_DELAY_SECONDS}s)")
    
    return requests._original_get(*args, **kwargs)

# Apply the patch
requests.get = safe_get
logging.info(f"🛡️ Safety Patch Applied: {SAFETY_DELAY_SECONDS}s delay enforced on all requests.")


def main():
    print("\n🏀 PortalRecruit | Golden Database Builder")
    print("===========================================")
    print("This script performs a 'Low & Slow' ingestion to build a static dataset.")
    print("Policy: Sequential requests, 2s delay, filtered scope.\n")

    # 1. Secure Input
    api_key = input("Enter Full Access Synergy API Key: ").strip()
    if not api_key:
        logging.error("No key provided.")
        return

    # 2. Discovery Phase
    print(f"\n🔍 Scanning for {CONFERENCE_FILTER} teams in {YEARS_TO_INGEST}...")
    report = discover_capabilities(api_key=api_key, league_code="ncaamb")
    
    target_season_ids = []
    target_team_ids = []

    # 3. Filtering Logic
    if not report.seasons:
        logging.error("No seasons found. Check API Key permissions.")
        return

    for season in report.seasons:
        # Match "2024" or "2025" in the season year or name
        if season.year in YEARS_TO_INGEST or any(str(y) in season.name for y in YEARS_TO_INGEST):
            target_season_ids.append(season.id)
            logging.info(f"✅ Found Target Season: {season.name} ({season.id})")
            
            # Find teams in this season belonging to the conference
            teams_in_season = report.teams_by_season.get(season.id, [])
            for team in teams_in_season:
                # Check Conference (Case insensitive)
                conf = (team.conference or "").lower()
                if CONFERENCE_FILTER.lower() in conf:
                    target_team_ids.append(team.id)
    
    # Deduplicate teams
    target_team_ids = list(set(target_team_ids))
    
    if not target_team_ids:
        logging.error(f"❌ No teams found for conference '{CONFERENCE_FILTER}'. Check spelling.")
        # Fallback: Ask user if they want to proceed with ALL teams (Dangerous, so we warn)
        confirm = input(f"Warning: No {CONFERENCE_FILTER} teams found. Proceed with ALL teams? (y/n): ")
        if confirm.lower() != 'y':
            return
    else:
        logging.info(f"✅ Identified {len(target_team_ids)} unique teams in {CONFERENCE_FILTER}.")

    # 4. Execution Loop
    # We run a separate pipeline for each season to keep checkpoints clean
    print(f"\n🚀 Starting Ingestion. Estimated time: {len(target_team_ids) * 30 * 2.0 / 60:.1f} minutes per season (approx).")
    print("DO NOT CLOSE THIS TERMINAL.\n")

    for season_id in target_season_ids:
        logging.info(f"📂 Processing Season ID: {season_id}")
        
        plan = PipelinePlan(
            league_code="ncaamb",
            season_id=season_id,
            team_ids=target_team_ids,
            ingest_events=True, # We need play-by-play for semantic search
        )

        try:
            # We don't need a progress_cb for CLI, logs handle it
            result = run_pipeline(plan=plan, api_key=api_key)
            logging.info(f"🎉 Season Complete. Games: {result.get('inserted_games')}, Plays: {result.get('inserted_plays')}")
        except Exception as e:
            logging.error(f"❌ Failed processing season {season_id}: {e}")

    print("\n===========================================")
    print("✅ Golden Database Build Complete.")
    print("You can now commit the 'chroma_db' (or equivalent) folder to Git.")

if __name__ == "__main__":
    main()
