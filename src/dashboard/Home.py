import sys
import streamlit as st
from pathlib import Path

# --- 1. SETUP PATHS ---
# Ensure repo root is on sys.path so imports work
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- 2. PAGE CONFIGURATION ---
WORDMARK_DARK_URL = "https://skoutsearch.github.io/PortalRecruit/PORTALRECRUIT_WORDMARK_DARK.jpg"
from src.dashboard.theme import inject_background

st.set_page_config(
    page_title="PortalRecruit | Search",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="expanded", # Changed to expanded for navigation
)
inject_background()

# --- 3. HELPER FUNCTIONS ---
def check_ingestion_status():
    """
    Checks if the vector database exists.
    """
    db_path = REPO_ROOT / "data" / "vector_db" / "chroma.sqlite3"
    return db_path.exists()

def render_header():
    st.markdown(
        f"""
        <div class="pr-hero">
          <img src="{WORDMARK_DARK_URL}" style="max-width:560px; width:min(560px, 92vw); height:auto; object-fit:contain;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- 4. MAIN APP LOGIC ---

# Initialize Session State for Navigation
if "app_mode" not in st.session_state:
    # Default to Search if data exists, otherwise send to Admin/Setup
    if check_ingestion_status():
        st.session_state.app_mode = "Search"
    else:
        st.session_state.app_mode = "Admin"

# Sidebar Navigation
with st.sidebar:
    st.header("Navigation")
    mode = st.radio(
        "Choose Mode:", 
        ["Search", "Admin"], 
        index=0 if st.session_state.app_mode == "Search" else 1
    )
    st.session_state.app_mode = mode
    st.divider()
    st.info(f"Current Status: {st.session_state.app_mode}")

# --- 5. RENDER CONTENT BASED ON MODE ---

if st.session_state.app_mode == "Admin":
    # ---------------- ADMIN / INGESTION VIEW ----------------
    render_header()
    st.caption("⚙️ Ingestion Pipeline & Settings")
    
    # Execute the existing admin_content.py
    admin_path = Path(__file__).with_name("admin_content.py")
    if admin_path.exists():
        code = admin_path.read_text(encoding="utf-8")
        exec(compile(code, str(admin_path), "exec"), globals(), globals())
    else:
        st.error(f"Could not find {admin_path}")

elif st.session_state.app_mode == "Search":
    # ---------------- SEARCH VIEW ----------------
    render_header()
    
    # This is where your Search UI lives. 
    # Ideally, put this in a separate file like `src/dashboard/search_ui.py` and import it.
    # For now, I'll place the logic block here.
    
    st.markdown("### 🔍 Semantic Player Search")

    # Advanced Filters (collapsed by default)
    min_dog = min_menace = min_unselfish = min_tough = min_rim = min_shot = 0
    n_results = 15
    tag_filter = []

    with st.sidebar.expander("Advanced Filters", expanded=False):
        min_dog = st.slider("Min Dog Index", 0, 100, 0)
        min_menace = st.slider("Min Defensive Menace", 0, 100, 0)
        min_unselfish = st.slider("Min Unselfishness", 0, 100, 0)
        min_tough = st.slider("Min Toughness", 0, 100, 0)
        min_rim = st.slider("Min Rim Pressure", 0, 100, 0)
        min_shot = st.slider("Min Shot Making", 0, 100, 0)
        n_results = st.slider("Number of Results", 5, 50, 15)
        tag_filter = st.multiselect(
            "Required Tags",
            ["drive", "rim_pressure", "pnr", "iso", "post_up", "handoff", "pull_up", "3pt", "jumpshot", "dunk", "layup", "steal", "block", "charge_taken", "loose_ball", "deflection", "assist", "turnover"],
            default=[]
        )

    query = st.chat_input("Describe the player you are looking for (e.g., 'A downhill guard who can guard')...")

    if query:
        # --- QUERY INTENTS (coach-speak -> filters) ---
        q = query.lower()
        if "get in the lane" in q or "rim pressure" in q or "downhill" in q:
            min_rim = max(min_rim, 30)
            tag_filter = list(set(tag_filter + ["drive", "rim_pressure"]))
        if "keep people out" in q or "rim protector" in q:
            min_menace = max(min_menace, 25)
            tag_filter = list(set(tag_filter + ["block"]))
        if "unselfish" in q or "selfless" in q:
            min_unselfish = max(min_unselfish, 40)
        if "tough" in q or "toughness" in q:
            min_tough = max(min_tough, 25)
            tag_filter = list(set(tag_filter + ["loose_ball", "charge_taken"]))
        if "shot maker" in q or "shot making" in q or "shooter" in q:
            min_shot = max(min_shot, 40)
            tag_filter = list(set(tag_filter + ["jumpshot", "3pt"]))
        if "defensive menace" in q or "menace" in q:
            min_menace = max(min_menace, 35)
            tag_filter = list(set(tag_filter + ["steal", "block", "deflection"]))

        st.write(f"Searching for: **{query}**")

        # --- VECTOR SEARCH ---
        import chromadb
        import sqlite3

        VECTOR_DB_PATH = REPO_ROOT / "data" / "vector_db"
        DB_PATH = REPO_ROOT / "data" / "skout.db"

        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        try:
            collection = client.get_collection(name="skout_plays")
        except Exception:
            st.error("Vector DB not found. Run embeddings first (generate_embeddings.py) to create 'skout_plays'.")
            st.stop()

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        play_ids = results.get("ids", [[]])[0]
        if not play_ids:
            st.warning("No results found.")
        else:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()

            # Pull plays
            placeholders = ",".join(["?"] * len(play_ids))
            cur.execute(
                f"""
                SELECT play_id, description, game_id, clock_display, player_id, player_name
                FROM plays
                WHERE play_id IN ({placeholders})
                """,
                play_ids,
            )
            play_rows = cur.fetchall()

            # Pull traits
            player_ids = [r[4] for r in play_rows if r[4]]
            traits = {}
            if player_ids:
                ph2 = ",".join(["?"] * len(set(player_ids)))
                cur.execute(
                    f"""
                    SELECT player_id, dog_index, menace_index, unselfish_index,
                           toughness_index, rim_pressure_index, shot_making_index
                    FROM player_traits
                    WHERE player_id IN ({ph2})
                    """,
                    list(set(player_ids)),
                )
                traits = {
                    r[0]: {
                        "dog": r[1],
                        "menace": r[2],
                        "unselfish": r[3],
                        "tough": r[4],
                        "rim": r[5],
                        "shot": r[6],
                    }
                    for r in cur.fetchall()
                }

            # Pull game matchup
            game_ids = list({r[2] for r in play_rows})
            matchups = {}
            if game_ids:
                ph3 = ",".join(["?"] * len(game_ids))
                cur.execute(
                    f"""
                    SELECT game_id, home_team, away_team, video_path
                    FROM games
                    WHERE game_id IN ({ph3})
                    """,
                    game_ids,
                )
                matchups = {r[0]: (r[1], r[2], r[3]) for r in cur.fetchall()}

            conn.close()

            # Build display rows (filter by dog index + tags)
            from src.processing.play_tagger import tag_play  # noqa: E402

            rows = []
            for pid, desc, gid, clock, player_id, player_name in play_rows:
                t = traits.get(player_id, {})
                dog_index = t.get("dog")
                menace_index = t.get("menace")
                unselfish_index = t.get("unselfish")
                tough_index = t.get("tough")
                rim_index = t.get("rim")
                shot_index = t.get("shot")

                if dog_index is not None and dog_index < min_dog:
                    continue
                if menace_index is not None and menace_index < min_menace:
                    continue
                if unselfish_index is not None and unselfish_index < min_unselfish:
                    continue
                if tough_index is not None and tough_index < min_tough:
                    continue
                if rim_index is not None and rim_index < min_rim:
                    continue
                if shot_index is not None and shot_index < min_shot:
                    continue

                play_tags = tag_play(desc)
                if "non_possession" in play_tags:
                    continue
                if tag_filter and not set(tag_filter).issubset(set(play_tags)):
                    continue

                home, away, video = matchups.get(gid, ("Unknown", "Unknown", None))
                rows.append({
                    "Matchup": f"{home} vs {away}",
                    "Clock": clock,
                    "Player": (player_name or "Unknown"),
                    "Dog Index": dog_index,
                    "Menace": menace_index,
                    "Unselfish": unselfish_index,
                    "Toughness": tough_index,
                    "Rim Pressure": rim_index,
                    "Shot Making": shot_index,
                    "Tags": ", ".join(play_tags),
                    "Play": desc,
                    "Video": video or "-",
                })

            if rows:
                st.markdown("### Results")
                for r in rows:
                    with st.container():
                        st.markdown(f"**{r['Player']}** — {r['Matchup']} @ {r['Clock']}")
                        st.caption(f"Tags: {r['Tags']}")
                        st.write(r["Play"])
                        if r.get("Video") and r["Video"] != "-":
                            try:
                                st.video(r["Video"])
                            except Exception:
                                pass
                        st.divider()
            else:
                st.info("No results after filters.")
