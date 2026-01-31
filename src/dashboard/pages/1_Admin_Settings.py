import streamlit as st
import os
import sys
import subprocess
from dotenv import load_dotenv

# Add project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../../'))
sys.path.append(PROJECT_ROOT)

# Page config is set by the single-page Home.py wrapper.

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
            f"**Teams list:** {'✅ available' if teams_ok else '❌ not available'} | **Games:** {'✅ available' if games_ok else '❌ not available'}"
        )

        selected_team_ids: list[str] = []
        if teams_ok and teams:
            # Sort teams so "unknown conference" is last; keep it readable for non-technical users.
            teams_sorted = sorted(
                teams,
                key=lambda t: (
                    (t.conference or "").lower() in {"unknown conference"},
                    (t.conference or "").lower(),
                    t.name.lower(),
                ),
            )

            team_names = [t.name for t in teams_sorted]
            team_id_by_name = {t.name: t.id for t in teams_sorted}

            selected_team_names = st.multiselect(
                "Teams (optional)",
                team_names,
                default=[],
                help="Tip: If you're unsure, leave this blank to ingest everything available for the season.",
            )
            selected_team_ids = [team_id_by_name[n] for n in selected_team_names]

            st.caption(
                "Next: run an end-to-end pipeline: schedule → events → (later: videos) → auto index."
            )
        elif teams_ok:
            st.info("Teams endpoint works, but no teams were returned for this season.")
        else:
            st.info("Teams list not available for this season with your current key.")

        st.markdown("---")
        st.subheader("Run Pipeline")
        st.caption("This is the new one-click path. It will replace the legacy buttons once stabilized.")

        ingest_events = st.toggle("Ingest Play-by-Play Events", value=True)

        from src.ingestion.pipeline import PipelinePlan, run_pipeline  # noqa: E402

        if st.button("Run Pipeline Now", type="primary"):
            if not api_key_for_scan:
                st.error("No API key available.")
            else:
                prog = st.progress(0)
                status = st.status("Starting pipeline...", expanded=True)

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
                        # map 55..90
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
                    status.update(label="Pipeline complete", state="complete")
                    prog.progress(100)
                    st.success(
                        f"Done. Games: {result['inserted_games']}, Plays: {result['inserted_plays']}"
                    )
                except Exception as e:
                    status.update(label="Pipeline failed", state="error")
                    st.exception(e)

    else:
        st.info("No seasons discovered yet. Your key may not allow listing seasons.")
