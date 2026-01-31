import streamlit as st
import os
import sys
import subprocess
from dotenv import load_dotenv

# Add project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../../'))
sys.path.append(PROJECT_ROOT)

st.set_page_config(page_title="PortalRecruit Admin", layout="wide", page_icon="⚙️")

ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

def save_local_api_key(key):
    """Writes the API key to the .env file (Local Dev Only)."""
    with open(ENV_PATH, "w") as f:
        f.write(f"SYNERGY_API_KEY={key}\n")
    os.environ["SYNERGY_API_KEY"] = key 
    st.toast("API Key Saved Locally!", icon="✅")

def run_ingestion_script(script_name, args=None):
    """Runs a python script as a subprocess."""
    script_path = os.path.join(PROJECT_ROOT, "src", script_name)
    args = args or []
    try:
        # We pass the current environment to the subprocess so it inherits secrets/env vars
        result = subprocess.run([sys.executable, script_path, *args], capture_output=True, text=True, env=os.environ)
        return result.stdout + "\n" + result.stderr
    except Exception as e:
        return str(e)

# --- LOAD SECRETS ---
# 1. Try Streamlit Cloud Secrets
cloud_key = st.secrets.get("SYNERGY_API_KEY", None)

# 2. Try Local .env
if not cloud_key:
    load_dotenv(ENV_PATH)
    local_key = os.getenv("SYNERGY_API_KEY", "")
else:
    local_key = cloud_key

st.title("⚙️ System Configuration")
st.markdown("Manage your Synergy Data connection and database status.")

# --- SECTION 1: API CREDENTIALS ---
st.subheader("1. Synergy API Connection")

if cloud_key:
    st.success("✅ Connected via Streamlit Cloud Secrets (Read-Only)")
    st.info("To change this key, update your settings in the Streamlit Cloud dashboard.")
else:
    with st.form("api_key_form"):
        user_key = st.text_input("Enter Synergy API Key", value=local_key, type="password")
        submit = st.form_submit_button("Save Credentials")
        
        if submit:
            save_local_api_key(user_key)
            st.rerun()

st.divider()

# --- SECTION 2: DATA ACCESS (DISCOVERY-FIRST) ---
st.subheader("2. Data Access (Discovery)")
st.caption("Scan your Synergy key to see exactly what data you can access, then select what you want to ingest.")

# Import here to keep Streamlit page load snappy
from src.ingestion.capabilities import discover_capabilities  # noqa: E402

api_key_for_scan = cloud_key or local_key

@st.cache_data(ttl=60 * 15, show_spinner=False)
def _cached_capabilities(api_key: str):
    # Avoid caching on full key string in clear: Streamlit cache is server-side, but we still
    # want to keep the object small. The api_key must still be used for requests.
    return discover_capabilities(api_key=api_key, league_code="ncaamb")

scan_col1, scan_col2 = st.columns([1, 3])
with scan_col1:
    do_scan = st.button("Scan API Access", use_container_width=True)
with scan_col2:
    st.info("This will probe seasons/teams/games with lightweight requests and report what your key can access.")

if do_scan:
    if not api_key_for_scan:
        st.error("No SYNERGY_API_KEY found. Add it via Streamlit Secrets or save it locally above.")
    else:
        with st.spinner("Scanning Synergy access..."):
            report = _cached_capabilities(api_key_for_scan)
        st.session_state["cap_report"] = report

report = st.session_state.get("cap_report")

if report:
    # Summary
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
        games_ok = report.games_accessible.get(chosen_season_id, False)

        st.markdown(
            f"**Teams endpoint:** {'✅' if teams_ok else '❌'} | **Games endpoint:** {'✅' if games_ok else '❌'}"
        )

        if teams_ok and teams:
            team_names = [t.name for t in teams]
            selected_teams = st.multiselect("Teams (optional)", team_names, default=[])
            st.caption(
                "Next: we'll use this selection to ingest schedule/events/videos + auto-build embeddings in one click."
            )
        elif teams_ok:
            st.info("Teams endpoint works, but no teams were returned for this season.")
        else:
            st.info("Teams list not available for this season with your current key.")

    else:
        st.info("No seasons discovered yet. Your key may not allow listing seasons.")

st.divider()

# --- SECTION 3: PIPELINE (LEGACY BUTTONS, TEMP) ---
st.subheader("3. Data Pipeline (Legacy)")
st.caption("These will be replaced by a single end-to-end pipeline button once the new selection UI is complete.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Step 1: Game Schedule**")
    st.info("Fetches games and caches them to the local SQLite DB.")
    if st.button("Sync Schedule"):
        with st.spinner("Fetching games..."):
            log = run_ingestion_script("ingestion/ingest_acc_schedule.py")
            st.text_area("Log", log, height=180)

with col2:
    st.markdown("**Step 2: Play Data**")
    st.info("Downloads PBP events for cached games.")
    if st.button("Sync Plays"):
        with st.spinner("Downloading plays..."):
            log = run_ingestion_script("ingestion/ingest_game_events.py")
            st.text_area("Log", log, height=180)

with col3:
    st.markdown("**Step 3: AI Intelligence**")
    st.info("Generates tags + embeddings.")
    if st.button("Build AI Index"):
        with st.spinner("Thinking..."):
            log1 = run_ingestion_script("processing/apply_tags.py")
            log2 = run_ingestion_script("processing/generate_embeddings.py")
            st.text_area("Log", log1 + "\n" + log2, height=180)
