import streamlit as st
import os
import sys
import subprocess
import time
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result

# --- 1. PROACTIVE RATE LIMITING (Monkey Patch) ---
# We overwrite requests.get globally so that src.ingestion.pipeline 
# automatically uses this robust version without needing code changes there.

# Save the original function so we can still call it
if not hasattr(requests, "_original_get"):
    requests._original_get = requests.get

def is_rate_limit_error(response):
    """Return True if we hit a 429 (Too Many Requests) or 500+ error."""
    if response.status_code == 429:
        print("⚠️ Rate limit hit (429). Retrying...")
        return True
    return response.status_code >= 500

@retry(
    retry=retry_if_result(is_rate_limit_error),
    wait=wait_exponential(multiplier=1.5, min=2, max=60), # Wait 2s, 3s, 4.5s...
    stop=stop_after_attempt(10)
)
def robust_get(*args, **kwargs):
    """
    A wrapper around requests.get that:
    1. Sleeps 1.1s BEFORE the request (Proactive limiting)
    2. Retries automatically if it hits a 429 (Reactive limiting)
    """
    time.sleep(1.1)  # Throttle speed: ~1 request per 1.1 seconds
    return requests._original_get(*args, **kwargs)

# Apply the patch
requests.get = robust_get


# --- 2. STANDARD SETUP ---

# Add project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../../'))
sys.path.append(PROJECT_ROOT)

ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

def save_local_api_key(key):
    """Writes the API key to the .env file (Local Dev Only)."""
    with open(ENV_PATH, "w") as f:
        f.write(f"SYNERGY_API_KEY={key}\n")
    os.environ["SYNERGY_API_KEY"] = key 
    st.toast("API Key Saved Locally!", icon="✅")

# --- LOAD SECRETS ---
try:
    cloud_key = st.secrets.get("SYNERGY_API_KEY", None)
except Exception:
    cloud_key = None

if not cloud_key:
    load_dotenv(ENV_PATH)
    local_key = os.getenv("SYNERGY_API_KEY", "")
else:
    local_key = cloud_key

# --- UI HEADER ---
st.markdown(
    """
<div style="text-align:center; margin: 6px 0 12px 0;">
  <div style="font-size:34px; font-weight:800;">System Configuration</div>
  <div style="opacity:0.74; font-size:14px;">Manage your Synergy data connection and database status.</div>
</div>
""",
    unsafe_allow_html=True,
)

# --- SECTION 1: API CREDENTIALS ---
st.markdown("<div style=\"font-size:20px; font-weight:800; margin-top:14px;\">1. Synergy API Connection</div>", unsafe_allow_html=True)

if cloud_key:
    st.success("✅ Connected via Streamlit Cloud Secrets (Read-Only)")
else:
    with st.form("api_key_form"):
        user_key = st.text_input("Enter Synergy API Key", value=local_key, type="password")
        submit = st.form_submit_button("Save Credentials")
        
        if submit:
            save_local_api_key(user_key)
            st.rerun()

st.divider()

# --- SECTION 2: DATA ACCESS ---
st.markdown("<div style=\"font-size:20px; font-weight:800; margin-top:14px;\">2. Data Access (Discovery)</div>", unsafe_allow_html=True)
st.caption("Scan your Synergy key to see exactly what data you can access.")

# Import modules AFTER patching requests
from src.ingestion.capabilities import discover_capabilities  # noqa: E402
from src.ingestion.pipeline import PipelinePlan, run_pipeline  # noqa: E402

api_key_for_scan = cloud_key or local_key

@st.cache_data(ttl=60 * 15, show_spinner=False)
def _cached_capabilities(api_key: str):
    return discover_capabilities(api_key=api_key, league_code="ncaamb")

scan_col1, scan_col2 = st.columns([1, 3])
with scan_col1:
    do_scan = st.button("Scan API Access", use_container_width=True)
with scan_col2:
    st.info("This will probe seasons/teams/games and report what your key can access.")

if do_scan:
    if not api_key_for_scan:
        st.error("No SYNERGY_API_KEY found.")
    else:
        # If local DB already exists + populated, guide user to search
        skout_db = os.path.join(PROJECT_ROOT, "data", "skout.db")
        if os.path.exists(skout_db) and os.path.getsize(skout_db) > 1024:
            st.success("✅ skout.db already exists. You can skip ingestion and go straight to Search.")
        with st.spinner("Scanning Synergy access (this may take a moment due to rate limiting)..."):
            report = _cached_capabilities(api_key_for_scan)
        st.session_state["cap_report"] = report

report = st.session_state.get("cap_report")

if report:
    # ... (Existing rendering logic for seasons/teams) ...
    ok_seasons = "✅" if report.seasons_accessible else "❌"
    st.markdown(f"**League:** `{report.league}`  ")
    st.markdown(f"**Seasons endpoint:** {ok_seasons}")

    if report.warnings:
        for w in report.warnings:
            st.warning(w)

    if report.seasons:
        season_labels = []
        season_id_by_label = {}
        for s in report.seasons:
            label = f"{s.year or ''} {s.name}".strip() or s.id
            season_labels.append(label)
            season_id_by_label[label] = s.id

        chosen_label = st.selectbox("Season", season_labels, index=0)
        chosen_season_id = season_id_by_label[chosen_label]
        
        teams = report.teams_by_season.get(chosen_season_id, [])
        teams_ok = report.teams_accessible.get(chosen_season_id, False)

        # Team sorting and selection logic
        import re
        def _pretty_team_name(name: str) -> str:
            n = (name or "").strip()
            if " " not in n and re.search(r"[a-z][A-Z]", n):
                n = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n)
            n = re.sub(r"\s+", " ", n).strip()
            return n

        if teams_ok and teams:
            teams_sorted = sorted(teams, key=lambda t: t.name.lower())
            
            option_labels = [f"{_pretty_team_name(t.name)} ({t.id})" for t in teams_sorted]
            team_id_map = {lbl: t.id for lbl, t in zip(option_labels, teams_sorted)}

            selected_labels = st.multiselect(
                "Teams (optional)",
                option_labels,
                default=[],
                help="Leave blank to ingest ALL teams.",
            )
            selected_team_ids = [team_id_map[lbl] for lbl in selected_labels]
        else:
            selected_team_ids = []

        st.markdown("---")
        st.markdown("<div style=\"font-size:20px; font-weight:800; margin-top:14px;\">3. Run Pipeline</div>", unsafe_allow_html=True)
        
        ingest_events = st.toggle("Ingest Play-by-Play Events", value=True)

        st.markdown("### 🚀 Jump to Search")
        if st.button("Open Search Interface"):
            st.session_state.app_mode = "Search"
            st.rerun()

        if st.button("Run Pipeline Now", type="primary"):
            if not api_key_for_scan:
                st.error("No API key available.")
            else:
                prog = st.progress(0)
                status = st.status("Starting pipeline...", expanded=True)

                # Check if Vector DB already exists
                vector_db_path = os.path.join(PROJECT_ROOT, "data", "vector_db", "chroma.sqlite3")
                if os.path.exists(vector_db_path):
                    st.info("Vector DB already exists. Skipping ingestion.")
                    prog.progress(100)
                else:
                    def _cb(step: str, info: dict):
                        if step == "schedule:start":
                            status.update(label="Ingesting schedule...", state="running")
                            prog.progress(10)
                        elif step == "schedule:done":
                            status.write(f"✅ Schedule cached: {info.get('inserted_games', 0)} games")
                            prog.progress(45)
                        elif step == "events:start":
                            status.update(label="Ingesting events...", state="running")
                            prog.progress(55)
                        elif step == "events:progress":
                            cur = info.get("current", 0)
                            total = max(1, info.get("total", 1))
                            pct = 55 + int(35 * (cur / total))
                            prog.progress(min(90, max(55, pct)))
                        elif step == "events:done":
                            status.write(f"✅ Events cached: {info.get('inserted_plays', 0)} plays")
                            prog.progress(95)

                    plan = PipelinePlan(
                        league_code="ncaamb",
                        season_id=chosen_season_id,
                        team_ids=selected_team_ids,
                        ingest_events=ingest_events,
                    )

                    try:
                        result = run_pipeline(plan=plan, api_key=api_key_for_scan, progress_cb=_cb)
                        status.write("✅ Ingestion complete. Running embeddings...")

                        # Run embeddings step to build Chroma DB
                        from src.processing.generate_embeddings import generate_embeddings  # noqa: E402
                        generate_embeddings()

                        status.write("✅ Backfilling player names...")
                        try:
                            from scripts.backfill_player_names import main as backfill_players  # noqa: E402
                            backfill_players()
                        except Exception as e:
                            status.write(f"⚠️ Backfill failed: {e}")

                        status.write("✅ Building player traits...")
                        try:
                            from src.processing.derive_player_traits import build_player_traits  # noqa: E402
                            build_player_traits()
                        except Exception as e:
                            status.write(f"⚠️ Traits build failed: {e}")

                        status.update(label="Pipeline complete", state="complete")
                        prog.progress(100)
                        
                        st.success(f"Success! Games: {result['inserted_games']}, Plays: {result['inserted_plays']}")
                        
                        # --- NEW: UNLOCK SEARCH ---
                        st.markdown("### 🚀 Ready to Search?")
                        if st.button("Open Search Interface"):
                            st.session_state.app_mode = "Search"
                            st.rerun()
                            
                    except Exception as e:
                        status.update(label="Pipeline failed", state="error")
                        st.exception(e)

    else:
        st.info("No seasons discovered yet.")
