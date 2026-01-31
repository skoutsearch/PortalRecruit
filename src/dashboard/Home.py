import streamlit as st
import sqlite3
import chromadb
import os
import sys
import re
import base64
from datetime import datetime

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
sys.path.append(PROJECT_ROOT) 

DB_PATH = os.path.join(PROJECT_ROOT, "data/skout.db")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "data/vector_db")
LOGO_PATH = os.path.join(PROJECT_ROOT, "www", "PORTALRECRUIT_LOGO.png")
BG_VIDEO_PATH = os.path.join(PROJECT_ROOT, "www", "PORTALRECRUIT_ANIMATED_LOGO.mp4")

# --- LOAD LEAGUE CONFIG ---
try:
    from config.ncaa_di_mens_basketball import NCAA_DI_MENS_BASKETBALL
    from config.ncaa_dii_mens_basketball import NCAA_DII_MENS_BASKETBALL
    from config.ncaa_diii_mens_basketball import NCAA_DIII_MENS_BASKETBALL
    LEAGUE_STRUCTURE = {
        "NCAA D1": NCAA_DI_MENS_BASKETBALL,
        "NCAA D2": NCAA_DII_MENS_BASKETBALL,
        "NCAA D3": NCAA_DIII_MENS_BASKETBALL
    }
except ImportError:
    LEAGUE_STRUCTURE = {}

# --- PAGE SETUP ---
st.set_page_config(
    page_title="PortalRecruit | Recruitment Engine", 
    layout="wide", 
    page_icon="🏀",
    initial_sidebar_state="expanded" 
)

# --- CUSTOM CSS ---
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_base64_video(video_path):
    if not video_path or not os.path.exists(video_path):
        return ""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def inject_custom_css():
    # Video background (preferred)
    video_encoded = get_base64_video(BG_VIDEO_PATH)

    # Fallback: solid background
    bg_style = ".stApp { background-color: #020617; }"

    video_html = ""
    if video_encoded:
        # A fixed video behind the app with a dark overlay for readability
        video_html = f"""
        <div class=\"bg-video-wrap\">
            <video class=\"bg-video\" autoplay loop muted playsinline>
                <source src=\"data:video/mp4;base64,{video_encoded}\" type=\"video/mp4\">
            </video>
            <div class=\"bg-video-overlay\"></div>
        </div>
        """

    st.markdown(f"""
    {video_html}
    <style>
        {bg_style}

        .bg-video-wrap {{
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            z-index: -1;
            pointer-events: none; /* critical: never block clicks */
        }}
        .bg-video-wrap * {
            pointer-events: none;
        }
        /* Keep Streamlit content above the video */
        .stApp {{
            position: relative;
            z-index: 0;
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
            /* "Apple" vibe: clean, subtle, non-distracting */
            filter: saturate(1.05) contrast(1.02) brightness(0.92) blur(3px);
        }}

        /* Minimal overlay: soft dark gradient + gentle vignette for legibility */
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
            pointer-events: none;
        }}

        /* Apple vibe: keep background clean & static (no animated glow) */

        h1, h2, h3, h4, h5, h6, p, div, span, label, li {{
            color: #f8fafc !important; 
            font-family: 'Inter', sans-serif;
        }}
        
        section[data-testid="stSidebar"] {{
            background: rgba(15, 23, 42, 0.60);
            backdrop-filter: blur(18px);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 0 0 1px rgba(255,255,255,0.025);
        }}
        
        .stTextInput > div > div > input, .stSelectbox > div > div > div, .stMultiSelect > div > div > div {{
            background-color: rgba(30, 41, 59, 0.6);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }}
        
        .streamlit-expanderHeader {{
            background-color: rgba(30, 41, 59, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            color: white !important;
        }}
        .streamlit-expanderContent {{
            background-color: rgba(15, 23, 42, 0.4) !important;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-top: none;
        }}
        
        .hero-card {{
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .step-container {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
            flex-wrap: wrap;
        }}
        .step-box {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            width: 220px;
            transition: transform 0.2s;
        }}
        .step-box:hover {{
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.07);
        }}
        .step-icon {{ font-size: 2.5rem; margin-bottom: 10px; }}
        .step-title {{ color: white; font-weight: bold; margin-bottom: 5px; font-size: 1.1rem; }}
        .step-desc {{ color: #94a3b8 !important; font-size: 0.9rem; }}
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- BACKEND FUNCTIONS ---
@st.cache_resource
def get_chroma_client():
    if not os.path.exists(VECTOR_DB_PATH):
        return None

    # --- CLOUD DEPLOYMENT FIX: Patch SQLite ---
    # Streamlit Cloud uses an old SQLite version incompatible with Chroma.
    # We patch it with pysqlite3-binary.
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        # Not on cloud or library missing
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
    parsed = []
    conn = get_database_connection()
    cursor = conn.cursor()

    if not results["ids"]:
        return []
    
    norm_selected_teams = {normalize_name(t) for t in selected_teams}

    for i, play_id in enumerate(results['ids'][0]):
        meta = results['metadatas'][0][i]
        cursor.execute("SELECT home_team, away_team, video_path, date FROM games WHERE game_id = ?", (meta['game_id'],))
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
            play_tags = meta['tags'].split(", ") if meta['tags'] else []
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

        parsed.append({
            "id": play_id,
            "matchup": f"{home} vs {away}",
            "desc": meta['original_desc'],
            "tags": meta['tags'],
            "clock": meta['clock'],
            "video": vid_path,
            "offset": offset,
            "score": results['distances'][0][i],
            "date": date_str
        })
    conn.close()
    return parsed

# --- SIDEBAR CONTENT ---
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)
else:
    st.sidebar.title("PortalRecruit 🏀")

st.sidebar.markdown("### 🔎 Filters")

# 1. Division
sel_div = st.sidebar.selectbox("Division", ["All"] + list(LEAGUE_STRUCTURE.keys()), index=1)

# 2. Conference (FIXED LOGIC)
available_conferences = []
if sel_div != "All":
    available_conferences = list(LEAGUE_STRUCTURE[sel_div].keys())

conf_options = ["All"] + sorted(available_conferences)
# If only "All" exists (length 1), force index 0. Otherwise try 1.
safe_conf_index = 1 if len(conf_options) > 1 else 0

sel_conf = st.sidebar.selectbox("Conference", conf_options, index=safe_conf_index)

# 3. Team
available_teams = []
if sel_conf != "All" and sel_div != "All":
    available_teams = LEAGUE_STRUCTURE[sel_div][sel_conf]
elif sel_div != "All":
    # Flatten everything in the division
    for conf_teams in LEAGUE_STRUCTURE[sel_div].values():
        available_teams.extend(conf_teams)

sel_teams = st.sidebar.multiselect("Team", sorted(available_teams), placeholder="Select Teams...")
sel_years = st.sidebar.slider("Season", 2020, datetime.now().year, (2020, datetime.now().year))

# --- MAIN CONTENT ---
st.title("PortalRecruit | Recruitment Engine")

# CHECK DB
db_exists = os.path.exists(DB_PATH)
if db_exists:
    conn = sqlite3.connect(DB_PATH)
    try:
        game_count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    except Exception:
        game_count = 0
    finally:
        conn.close()
else:
    game_count = 0

# --- HERO INSTRUCTIONS ---
if game_count == 0:
    st.markdown("""
<div class="hero-card">
    <h2 style="font-weight: 800; font-size: 2.2rem; margin-bottom: 10px;">👋 Welcome to PortalRecruit</h2>
    <p style="color: #cbd5e1 !important; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
        Your recruitment engine is online. The database is currently empty. <br>
        Follow these three steps to initialize the system.
    </p>
    <div class="step-container">
        <div class="step-box">
            <div class="step-icon">🔑</div>
            <div class="step-title">1. Credentials</div>
            <div class="step-desc">Go to <b>Admin Settings</b> and enter your Synergy API Key.</div>
        </div>
        <div class="step-box">
            <div class="step-icon">⬇️</div>
            <div class="step-title">2. Ingest</div>
            <div class="step-desc">Click <b>Sync Schedule</b> and then <b>Sync Plays</b>.</div>
        </div>
        <div class="step-box">
            <div class="step-icon">🧠</div>
            <div class="step-title">3. Index</div>
            <div class="step-desc">Click <b>Build AI Index</b> to activate semantic search.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# --- SEARCH ---
col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input("Semantic Search", placeholder="e.g. 'Freshman turnovers', 'Pick and roll lob'")
with col2:
    real_tags = get_unique_tags()
    selected_tags_filter = st.multiselect("Tags", real_tags, placeholder="Add tags...")

# --- RESULTS ---
if search_query or selected_tags_filter:
    st.divider()
    with st.spinner("Analyzing tape..."):
        results = search_plays(search_query, selected_tags_filter, sel_teams, sel_years)
    
    if not results:
        st.warning("No plays found matching criteria.")
    else:
        st.success(f"Found {len(results)} plays.")
        for idx, play in enumerate(results):
            with st.container():
                label = f"{idx+1}. {play['matchup']} ({play['date']}) | ⏰ {play['clock']}"
                with st.expander(label, expanded=(idx == 0)):
                    c1, c2 = st.columns([1.2, 2])
                    with c1:
                        st.markdown(f"**{play['desc']}**")
                        if play['tags']:
                            tags = play['tags'].split(", ")
                            chips = ""
                            for t in tags:
                                color = "blue"
                                if "turnover" in t:
                                    color = "red"
                                if "made" in t or "score" in t:
                                    color = "green"
                                chips += f":{color}[`{t}`] "
                            st.markdown(chips)
                    with c2:
                        if play['video']:
                            st.video(play['video'], start_time=int(play['offset']))
                        else:
                            st.info("Video unavailable.")
