import streamlit as st
import sqlite3
import chromadb
import os
import sys
import re
# (base64 removed; we now load background video via GitHub Pages URL)
from datetime import datetime

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.append(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "skout.db")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "data", "vector_db")
LOGO_PATH = os.path.join(PROJECT_ROOT, "www", "PORTALRECRUIT_LOGO.png")
# Serve background video from GitHub Pages (fast, avoids base64 embedding in Streamlit)
BG_VIDEO_URL = "https://skoutsearch.github.io/PortalRecruit/PORTALRECRUIT_ANIMATED_LOGO.mp4"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="PortalRecruit | Recruitment Engine",
    layout="wide",
    page_icon="🏀",
    # We want maximum canvas for the stepper/progress UI; the flow lives in-page.
    initial_sidebar_state="collapsed",
)

# --- CUSTOM CSS ---
def inject_custom_css() -> None:
    # Fallback: solid background
    bg_style = ".stApp { background-color: #020617; }"

    # Host the background video on GitHub Pages so Streamlit doesn't have to embed base64.
    video_html = f"""
    <div class=\"bg-video-wrap\">
        <video class=\"bg-video\" autoplay loop muted playsinline>
            <source src=\"{BG_VIDEO_URL}\" type=\"video/mp4\">
        </video>
        <div class=\"bg-video-overlay\"></div>
    </div>
    """

    st.markdown(
        f"""
    {video_html}
    <style>
        {bg_style}

        .bg-video-wrap {{
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            z-index: 0; /* avoid negative z-index quirks */
            pointer-events: none; /* critical: never block clicks */
        }}
        .bg-video-wrap * {{
            pointer-events: none;
        }}
        .stApp {{
            position: relative;
            z-index: 1; /* ensure UI paints above background */
        }}

        .bg-video {{
            position: absolute;
            top: 50%;
            left: 50%;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            transform: translate(-50%, -50%) scale(1.03);
            object-fit: cover;
            opacity: 0.32;
            filter: saturate(1.05) contrast(1.02) brightness(0.92) blur(3px);
        }}

        .bg-video-overlay {{
            position: absolute;
            inset: 0;
            background: linear-gradient(
              180deg,
              rgba(2, 6, 23, 0.88) 0%,
              rgba(2, 6, 23, 0.82) 45%,
              rgba(2, 6, 23, 0.90) 100%
            );
        }}
        .bg-video-overlay::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(closest-side, rgba(0,0,0,0) 62%, rgba(0,0,0,0.40) 100%);
        }}

        h1, h2, h3, h4, h5, h6, p, div, span, label, li {{
            color: #f8fafc !important;
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        }}

        .glass-card {{
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 18px;
            padding: 22px;
            backdrop-filter: blur(18px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }}

        .muted {{
            color: rgba(226, 232, 240, 0.78) !important;
        }}

        /* Keep expanders consistent with the aesthetic */
        div[data-testid="stExpander"] > details {{
            background: rgba(15, 23, 42, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            overflow: hidden;
        }}

        /* This app is single-page. Hide the multipage sidebar/nav completely. */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarNav"],
        button[data-testid="collapsedControl"] {{
            display: none !important;
        }}

    </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

# --- BACKEND FUNCTIONS (search) ---
@st.cache_resource
def get_chroma_client():
    if not os.path.exists(VECTOR_DB_PATH):
        return None

    # Streamlit Cloud SQLite patch for Chroma
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

    return chromadb.PersistentClient(path=VECTOR_DB_PATH)


def get_database_connection():
    return sqlite3.connect(DB_PATH)


def get_unique_tags():
    if not os.path.exists(DB_PATH):
        return []

    conn = get_database_connection()
    try:
        rows = conn.execute("SELECT tags FROM plays WHERE tags != ''").fetchall()
        unique_tags: set[str] = set()
        for r in rows:
            tags = [t.strip() for t in r[0].split(",")]
            unique_tags.update(tags)
        return sorted(unique_tags)
    except Exception:
        return []
    finally:
        conn.close()


def normalize_name(name):
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def search_plays(query, selected_tags, selected_teams, year_range, n_results=50):
    client = get_chroma_client()
    if not client:
        return []

    try:
        collection = client.get_collection(name="skout_plays")
    except Exception:
        return []

    search_text = query if query else " ".join(selected_tags)
    if not search_text:
        return []

    results = collection.query(query_texts=[search_text], n_results=n_results)
    if not results.get("ids"):
        return []

    parsed = []
    conn = get_database_connection()
    cursor = conn.cursor()

    norm_selected_teams = {normalize_name(t) for t in selected_teams}

    for i, play_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        cursor.execute(
            "SELECT home_team, away_team, video_path, date FROM games WHERE game_id = ?",
            (meta["game_id"],),
        )
        game = cursor.fetchone()
        if not game:
            continue

        home, away, vid_path, date_str = game

        if norm_selected_teams:
            if normalize_name(home) not in norm_selected_teams and normalize_name(away) not in norm_selected_teams:
                continue

        game_year = int(date_str[:4]) if date_str else 0
        if game_year < year_range[0] or game_year > year_range[1]:
            continue

        if selected_tags:
            play_tags = meta.get("tags", "").split(", ") if meta.get("tags") else []
            if not all(tag in play_tags for tag in selected_tags):
                continue

        period_len = 1200
        cursor.execute("SELECT period, clock_seconds FROM plays WHERE play_id = ?", (play_id,))
        p_row = cursor.fetchone()
        offset = 0
        if p_row:
            period, clock_sec = p_row
            if period == 1:
                offset = max(0, period_len - clock_sec)
            elif period == 2:
                offset = max(0, 1200 + (period_len - clock_sec))

        parsed.append(
            {
                "id": play_id,
                "matchup": f"{home} vs {away}",
                "desc": meta.get("original_desc"),
                "tags": meta.get("tags"),
                "clock": meta.get("clock"),
                "video": vid_path,
                "offset": offset,
                "score": results["distances"][0][i],
                "date": date_str,
            }
        )

    conn.close()
    return parsed


# --- ONE-PAGE FLOW STATE ---
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "cap_report" not in st.session_state:
    st.session_state.cap_report = None
if "plan" not in st.session_state:
    st.session_state.plan = None
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "ready_to_search" not in st.session_state:
    st.session_state.ready_to_search = False


# --- HEADER ---
left, right = st.columns([1, 5])
with left:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=96)
with right:
    st.markdown("# PortalRecruit")
    st.markdown(
        "<div class='muted'>Semantic search across game tape — built from whatever your Synergy key can access.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# Determine stage
api_key = st.session_state.api_key
cap_report = st.session_state.cap_report
plan = st.session_state.plan
pipeline_result = st.session_state.pipeline_result

stage = 0
if api_key:
    stage = 1
if cap_report:
    stage = 2
if plan is not None:
    stage = 3
if pipeline_result is not None:
    stage = 4
if st.session_state.ready_to_search:
    stage = 5

# Imports for discovery/pipeline
from src.ingestion.capabilities import discover_capabilities  # noqa: E402
from src.ingestion.pipeline import PipelinePlan, run_pipeline  # noqa: E402


# STEP 0: API Key
with st.expander("Step 1 — Add your Synergy API key", expanded=(stage == 0)):
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Connect your data")
    st.markdown(
        "Enter your Synergy API key. We use it to discover what data you can access and build your search index.",
        unsafe_allow_html=True,
    )

    with st.form("api_key_form"):
        key_val = st.text_input("Synergy API Key", value=api_key, type="password")
        submitted = st.form_submit_button("Save & Continue")

    if submitted:
        st.session_state.api_key = key_val.strip()
        st.session_state.cap_report = None
        st.session_state.plan = None
        st.session_state.pipeline_result = None
        st.session_state.ready_to_search = False
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# STEP 1: Discover access
with st.expander("Step 2 — Discover what your key can access", expanded=(stage == 1)):
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    if not api_key:
        st.info("Add your API key above to begin.")
    else:
        st.markdown("### Scanning access")
        st.markdown(
            "We’ll probe seasons, teams, and games endpoints with lightweight requests.",
            unsafe_allow_html=True,
        )

        if st.button("Scan API Access", type="primary"):
            with st.status("Scanning Synergy access…", expanded=True) as status:
                status.write("Checking seasons…")
                report = discover_capabilities(api_key=api_key, league_code="ncaamb")
                status.write("Compiling access report…")
                status.update(label="Access scan complete", state="complete")

            st.session_state.cap_report = report
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# STEP 2: Selection
with st.expander("Step 3 — Choose what to ingest", expanded=(stage == 2)):
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    if not cap_report:
        st.info("Run the access scan first.")
    else:
        st.markdown("### Available options")
        if cap_report.warnings:
            for w in cap_report.warnings:
                st.warning(w)

        if not cap_report.seasons:
            st.error("No seasons discovered for this key. We’ll add fallback probing next.")
        else:
            labels = []
            season_id_by_label = {}
            for s in cap_report.seasons:
                label = f"{s.year or ''} {s.name}".strip() or s.id
                labels.append(label)
                season_id_by_label[label] = s.id

            chosen_label = st.selectbox("Season", labels, index=0)
            chosen_season_id = season_id_by_label[chosen_label]

            teams = cap_report.teams_by_season.get(chosen_season_id, [])
            team_ids: list[str] = []
            if teams:
                team_name_to_id = {t.name: t.id for t in teams}
                team_names = list(team_name_to_id.keys())
                selected_team_names = st.multiselect("Teams (optional)", team_names, default=[])
                team_ids = [team_name_to_id[n] for n in selected_team_names]
            else:
                st.info("Teams list not available (or empty). We’ll ingest whatever the games endpoint allows.")

            ingest_events = st.toggle("Also ingest Play-by-Play events", value=True)

            # quick, honest ETA guess (we’ll refine)
            est_games = 350 if not team_ids else min(400, max(35, 35 * len(team_ids)))
            est_minutes_low = max(1, int(est_games / 120))
            est_minutes_high = max(2, int(est_games / 40))
            st.caption(f"Estimate: ~{est_minutes_low}–{est_minutes_high} minutes (rough).")

            if st.button("Continue to confirmation", type="primary"):
                st.session_state.plan = {
                    "league_code": "ncaamb",
                    "season_id": chosen_season_id,
                    "team_ids": team_ids,
                    "ingest_events": ingest_events,
                }
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# STEP 3: Confirm
with st.expander("Step 4 — Confirm & run", expanded=(stage == 3)):
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    if not plan:
        st.info("Complete the selection step first.")
    else:
        st.markdown("### Confirm")
        st.code(plan, language="json")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Back", use_container_width=True):
                st.session_state.plan = None
                st.rerun()
        with c2:
            if st.button("Run Ingestion", type="primary", use_container_width=True):
                prog = st.progress(0)
                status = st.status("Starting pipeline…", expanded=True)

                def _cb(step: str, info: dict):
                    if step == "schedule:start":
                        status.update(label="Ingesting schedule…", state="running")
                        prog.progress(10)
                    elif step == "schedule:done":
                        status.write(f"✅ Schedule cached: {info.get('inserted_games', 0)} games")
                        prog.progress(45)
                    elif step == "events:start":
                        status.update(label="Ingesting events…", state="running")
                        prog.progress(55)
                    elif step == "events:progress":
                        cur = info.get("current", 0)
                        total = max(1, info.get("total", 1))
                        pct = 55 + int(35 * (cur / total))
                        prog.progress(min(90, max(55, pct)))
                    elif step == "events:done":
                        status.write(f"✅ Events cached: {info.get('inserted_plays', 0)} plays")
                        prog.progress(95)

                plan_obj = PipelinePlan(
                    league_code=plan["league_code"],
                    season_id=plan["season_id"],
                    team_ids=plan["team_ids"],
                    ingest_events=plan["ingest_events"],
                )

                try:
                    result = run_pipeline(plan=plan_obj, api_key=api_key, progress_cb=_cb)
                    status.update(label="Pipeline complete", state="complete")
                    prog.progress(100)
                    st.session_state.pipeline_result = result
                    st.rerun()
                except Exception as e:
                    status.update(label="Pipeline failed", state="error")
                    st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)


# STEP 4: Ready
with st.expander("Step 5 — Start searching", expanded=(stage >= 4)):
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    if pipeline_result is None:
        st.info("Run ingestion first.")
    else:
        st.success(
            f"Database populated. Games: {pipeline_result.get('inserted_games', 0)}, Plays: {pipeline_result.get('inserted_plays', 0)}"
        )
        st.markdown(
            "When you’re ready, jump into semantic search. Try queries like:"
            "\n- **'press break'**\n- **'high motor rebounder'**\n- **'PnR coverage breakdown'**",
        )

        if st.button("Start Searching", type="primary"):
            st.session_state.ready_to_search = True
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# SEARCH UI (only after user clicks Start Searching)
if st.session_state.ready_to_search:
    st.markdown("---")
    st.markdown("## Search")

    # Minimal filters (we’ll evolve these as we normalize metadata)
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Semantic Search",
            placeholder="e.g. 'Freshman turnovers', 'Pick and roll lob'",
        )
    with col2:
        real_tags = get_unique_tags()
        selected_tags_filter = st.multiselect("Tags", real_tags, placeholder="Add tags…")

    # Year filter
    sel_years = st.slider("Season (year)", 2020, datetime.now().year, (2020, datetime.now().year))

    if search_query or selected_tags_filter:
        st.divider()
        with st.spinner("Analyzing tape…"):
            results = search_plays(search_query, selected_tags_filter, selected_teams=[], year_range=sel_years)

        if not results:
            st.warning("No plays found matching criteria.")
        else:
            st.success(f"Found {len(results)} plays.")
            for idx, play in enumerate(results):
                label = f"{idx+1}. {play['matchup']} ({play['date']}) | ⏰ {play['clock']}"
                with st.expander(label, expanded=(idx == 0)):
                    c1, c2 = st.columns([1.2, 2])
                    with c1:
                        st.markdown(f"**{play['desc']}**")
                        if play.get("tags"):
                            tags = play["tags"].split(", ")
                            chips = " ".join([f"`{t}`" for t in tags])
                            st.markdown(chips)
                    with c2:
                        if play.get("video"):
                            st.video(play["video"], start_time=int(play["offset"]))
                        else:
                            st.info("Video unavailable.")
