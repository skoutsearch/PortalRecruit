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
    min_size = 0
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

    # Search box + autocomplete
    query = st.text_input("Search", "", placeholder="Downhill guard who can guard")

    # Recent searches
    try:
        import json
        mem_path = REPO_ROOT / "data" / "search_memory.json"
        if mem_path.exists():
            memory = json.loads(mem_path.read_text())
            recent = list(reversed(memory.get("queries", [])))[:8]
            if recent:
                recent_pick = st.selectbox("Recent searches", ["(none)"] + recent, index=0)
                if recent_pick != "(none)":
                    query = recent_pick
    except Exception:
        pass

    from src.search.autocomplete import suggest_rich  # noqa: E402
    suggestions = suggest_rich(query, limit=25)
    if suggestions:
        picked = st.selectbox("Suggestions", ["(keep typing)"] + suggestions, index=0)
        if picked != "(keep typing)":
            query = picked

    if query:
        # --- QUERY INTENTS (coach-speak -> filters) ---
        from src.search.coach_dictionary import infer_intents_verbose, INTENTS  # noqa: E402

        intents = infer_intents_verbose(query)
        exclude_tags = set()
        role_hints = set()
        explain = []
        matched_phrases = []

        for hit, phrase in intents.values():
            intent = hit.intent
            w = hit.weight
            role_hints |= hit.role_hints
            matched_phrases.append(phrase)
            explain.append(f"Matched '{phrase}' → {list(intent.traits.keys())}")

            min_dog = max(min_dog, int(intent.traits.get("dog", 0) * w))
            min_menace = max(min_menace, int(intent.traits.get("menace", 0) * w))
            min_unselfish = max(min_unselfish, int(intent.traits.get("unselfish", 0) * w))
            min_tough = max(min_tough, int(intent.traits.get("tough", 0) * w))
            min_rim = max(min_rim, int(intent.traits.get("rim", 0) * w))
            min_shot = max(min_shot, int(intent.traits.get("shot", 0) * w))
            # size/measurables intent triggers
            if intent is INTENTS.get("size_measurables"):
                min_size = max(min_size, 70)
            tag_filter = list(set(tag_filter + list(intent.tags)))
            exclude_tags |= intent.exclude_tags

        # Role hints to lightly nudge tags
        if "guard" in role_hints:
            tag_filter = list(set(tag_filter + ["drive", "pnr"]))
        if "wing" in role_hints:
            tag_filter = list(set(tag_filter + ["3pt", "deflection"]))
        if "big" in role_hints:
            tag_filter = list(set(tag_filter + ["rim_protection", "post_up"]))

        st.write(f"Searching for: **{query}**")
        if explain:
            st.caption("Why these results: " + " | ".join(explain))

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

        # Query expansion: add matched phrases to help retrieval
        expanded_query = query
        if matched_phrases:
            expanded_query = query + " | " + " | ".join(matched_phrases)

        results = collection.query(
            query_texts=[expanded_query],
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
                           toughness_index, rim_pressure_index, shot_making_index, size_index
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
                        "size": r[7],
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
                if t.get("size") is not None and t.get("size") < min_size:
                    continue

                play_tags = tag_play(desc)
                if "non_possession" in play_tags:
                    continue
                if exclude_tags and set(play_tags).intersection(exclude_tags):
                    continue
                if tag_filter and not set(tag_filter).issubset(set(play_tags)):
                    continue

                # Rerank score: trait alignment + tag matches
                score = 0
                for key, weight in [
                    ("dog", 0.5),
                    ("menace", 0.7),
                    ("unselfish", 0.6),
                    ("tough", 0.6),
                    ("rim", 0.7),
                    ("shot", 0.7),
                ]:
                    val = t.get(key) or 0
                    score += val * weight
                score += len(set(play_tags).intersection(set(tag_filter))) * 10

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
                    "Score": round(score, 2),
                })

            rows.sort(key=lambda r: r.get("Score", 0), reverse=True)

            # --- Search memory ---
            try:
                mem_path = REPO_ROOT / "data" / "search_memory.json"
                mem_path.parent.mkdir(parents=True, exist_ok=True)
                import json
                if mem_path.exists():
                    memory = json.loads(mem_path.read_text())
                else:
                    memory = {"queries": []}
                if query:
                    memory["queries"].append(query)
                    memory["queries"] = memory["queries"][-20:]
                    mem_path.write_text(json.dumps(memory, indent=2))
            except Exception:
                pass

            if rows:
                # Group by player (top 3 clips each)
                grouped = {}
                for r in rows:
                    grouped.setdefault(r["Player"], []).append(r)

                st.markdown("### Results")

                for player, clips in grouped.items():
                    st.markdown(f"## {player}")
                    for r in clips[:3]:
                        with st.container():
                            st.markdown(f"**{r['Matchup']}** @ {r['Clock']}")
                            st.caption(f"Tags: {r['Tags']} | Score: {r.get('Score', 0)}")
                            st.write(r["Play"])
                            if r.get("Video") and r["Video"] != "-":
                                try:
                                    st.video(r["Video"])
                                except Exception:
                                    pass
                            st.divider()
            else:
                st.info("No results after filters.")
