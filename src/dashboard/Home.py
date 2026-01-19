import streamlit as st
import sqlite3
import chromadb
import os
import sys
from datetime import datetime

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
DB_PATH = os.path.join(PROJECT_ROOT, "data/skout.db")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "data/vector_db")
LOGO_PATH = os.path.join(PROJECT_ROOT, "www", "SKOUT_LOGO.png")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SKOUT | Recruitment Engine", 
    layout="wide", 
    page_icon="🏀",
    initial_sidebar_state="expanded"
)

# --- BACKEND FUNCTIONS ---
@st.cache_resource
def get_chroma_client():
    if not os.path.exists(VECTOR_DB_PATH):
        return None
    return chromadb.PersistentClient(path=VECTOR_DB_PATH)

def get_database_connection():
    return sqlite3.connect(DB_PATH)

def get_unique_tags():
    """Scans the database to find every unique tag available."""
    if not os.path.exists(DB_PATH): return []
    conn = get_database_connection()
    try:
        # Get all tag strings (e.g., "pnr, missed, 3pt")
        rows = conn.execute("SELECT tags FROM plays WHERE tags != ''").fetchall()
        unique_tags = set()
        for r in rows:
            # Split "pnr, missed" -> ["pnr", " missed"] -> clean up
            tags = [t.strip() for t in r[0].split(",")]
            unique_tags.update(tags)
        return sorted(list(unique_tags))
    except:
        return []
    finally:
        conn.close()

def get_hierarchy_data():
    """Fetches Teams and Years for the dropdown filters."""
    if not os.path.exists(DB_PATH): return [], []
    conn = get_database_connection()
    try:
        teams = conn.execute("SELECT DISTINCT home_team FROM games UNION SELECT DISTINCT away_team FROM games").fetchall()
        dates = conn.execute("SELECT MIN(date), MAX(date) FROM games").fetchone()
        
        team_list = sorted([t[0] for t in teams])
        
        # Parse Dates (Format: YYYY-MM-DD)
        min_year = int(dates[0][:4]) if dates and dates[0] else 2020
        max_year = int(dates[1][:4]) if dates and dates[1] else datetime.now().year
        
        return team_list, (min_year, max_year)
    except:
        return [], (2020, 2025)
    finally:
        conn.close()

def calculate_video_offset(period, clock_seconds):
    # Estimator for condensed games
    period_length = 1200 
    if period == 1: return max(0, period_length - clock_seconds)
    elif period == 2: return max(0, 1200 + (period_length - clock_seconds))
    return 0

def search_plays(query, selected_tags, selected_teams, year_range, n_results=50):
    client = get_chroma_client()
    if not client: return []
    
    try:
        collection = client.get_collection(name="skout_plays")
    except:
        return []

    # 1. QUERY VECTOR DB
    # If user provided text, use it. If only tags, use tags as the query text.
    search_text = query if query else " ".join(selected_tags)
    if not search_text: return []

    results = collection.query(query_texts=[search_text], n_results=n_results)
    
    parsed = []
    conn = get_database_connection()
    cursor = conn.cursor()

    if not results['ids']: return []

    for i, play_id in enumerate(results['ids'][0]):
        meta = results['metadatas'][0][i]
        game_id = meta['game_id']
        
        # 2. FETCH GAME CONTEXT
        cursor.execute("SELECT home_team, away_team, video_path, date FROM games WHERE game_id = ?", (game_id,))
        game = cursor.fetchone()
        
        if not game: continue
        
        home, away, vid_path, date_str = game
        
        # --- 3. APPLY FILTERS (Python Side) ---
        
        # A. Team Filter
        if selected_teams:
            if home not in selected_teams and away not in selected_teams:
                continue
                
        # B. Year Filter
        game_year = int(date_str[:4]) if date_str else 0
        if game_year < year_range[0] or game_year > year_range[1]:
            continue

        # C. Tag Filter (Strict: Play MUST have ALL selected tags)
        if selected_tags:
            play_tags = meta['tags'].split(", ") if meta['tags'] else []
            # Check if all selected tags are present in the play's tags
            if not all(tag in play_tags for tag in selected_tags):
                continue

        # Calculate Offset
        cursor.execute("SELECT period, clock_seconds FROM plays WHERE play_id = ?", (play_id,))
        p_row = cursor.fetchone()
        offset = calculate_video_offset(p_row[0], p_row[1]) if p_row else 0

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

# --- GUI LAYOUT ---

# 1. BRANDING
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)
else:
    st.sidebar.header("SKOUT 🏀")

# 2. SIDEBAR FILTERS
st.sidebar.markdown("### 🔎 Scouting Filters")

# Hierarchy Filters
# Note: Since we only have ACC data currently, we hardcode the top levels for the demo feel.
# In a full production version, these would query the 'leagues' table.
sel_div = st.sidebar.selectbox("Division", ["All", "NCAA D1", "NCAA D2", "NBA"], index=1)
sel_conf = st.sidebar.selectbox("Conference", ["All", "ACC", "SEC", "Big 10", "Big 12"], index=1)

# Dynamic Filters
all_teams, (min_y, max_y) = get_hierarchy_data()
sel_teams = st.sidebar.multiselect("Teams", all_teams, placeholder="Select specific teams...")
sel_years = st.sidebar.slider("Season Range", min_y, max_y, (min_y, max_y))

st.sidebar.divider()
st.sidebar.caption(f"Connected to SKOUT Brain v1.0")

# 3. MAIN AREA
st.title("SKOUT | Recruitment Engine")

# --- ONBOARDING / EMPTY STATE CHECK ---
if not all_teams:
    st.info("👋 **Welcome to SKOUT!** Your database is currently empty.")
    
    with st.expander("🚀 Getting Started Guide (Read Me)", expanded=True):
        st.markdown("""
        ### How to populate your engine:
        1. **Get your Synergy API Key**: You will need a valid `Client ID` or `API Key` from Synergy Sports.
        2. **Go to Admin Settings**: Click **Admin_Settings** in the left sidebar.
        3. **Enter Credentials**: Paste your key into the secure form.
        4. **Sync Data**: Click "Sync Schedule" then "Sync Plays" to pull the latest D1 data.
        5. **Build AI**: Click "Build AI Index" to make the data searchable.
        """)
        st.stop() # Stop rendering the rest of the app until data exists

# 4. SEARCH INTERFACE
col1, col2 = st.columns([3, 1])

with col1:
    # Text Search
    search_query = st.text_input("Semantic Search", placeholder="e.g. 'Freshman turnovers', 'Pick and roll lob', 'Transition defense'")

with col2:
    # Tag Dropdown
    available_tags = get_unique_tags()
    selected_tags = st.multiselect("Filter by Tags", available_tags, placeholder="Select tags...")

# 5. RESULTS AREA
if search_query or selected_tags:
    st.divider()
    
    with st.spinner("Analyzing game tape..."):
        results = search_plays(search_query, selected_tags, sel_teams, sel_years)
    
    if not results:
        st.warning("No plays found matching your criteria.")
    else:
        st.success(f"Found {len(results)} plays matching criteria.")
        
        for idx, play in enumerate(results):
            # Card Styling
            with st.container():
                label = f"{idx+1}. {play['matchup']} ({play['date']}) | ⏰ {play['clock']}"
                
                with st.expander(label, expanded=(idx == 0)):
                    c1, c2 = st.columns([1, 1.5])
                    
                    with c1:
                        st.markdown(f"**Play:** {play['desc']}")
                        
                        # Display Tags as Chips
                        if play['tags']:
                            tags = play['tags'].split(", ")
                            # Color code specific tags
                            chips = ""
                            for t in tags:
                                color = "blue"
                                if "turnover" in t: color = "red"
                                if "made" in t: color = "green"
                                chips += f":{color}[`{t}`] "
                            st.markdown(chips)
                            
                        st.caption(f"AI Confidence: {play['score']:.4f}")
                        
                    with c2:
                        if play['video']:
                            st.video(play['video'], start_time=int(play['offset']))
                        else:
                            st.error("Video source unavailable")
