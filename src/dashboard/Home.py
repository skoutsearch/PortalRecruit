import sys
import streamlit as st
from pathlib import Path

# --- 1. SETUP PATHS ---
# Ensure repo root is on sys.path so imports work
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- 2. PAGE CONFIGURATION ---
WORDMARK_DARK_URL = "https://skoutsearch.github.io/PortalRecruit/PORTALRECRUIT_WORDMARK_LIGHT.png"
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
    slider_dog = slider_menace = slider_unselfish = slider_tough = slider_rim = slider_shot = 0
    slider_size = 0
    intent_dog = intent_menace = intent_unselfish = intent_tough = intent_rim = intent_shot = 0
    intent_size = 0
    n_results = 15
    tag_filter = []
    intent_tags = []
    required_tags = []

    with st.sidebar.expander("Advanced Filters", expanded=False):
        slider_dog = st.slider("Min Dog Index", 0, 100, 0)
        slider_menace = st.slider("Min Defensive Menace", 0, 100, 0)
        slider_unselfish = st.slider("Min Unselfishness", 0, 100, 0)
        slider_tough = st.slider("Min Toughness", 0, 100, 0)
        slider_rim = st.slider("Min Rim Pressure", 0, 100, 0)
        slider_shot = st.slider("Min Shot Making", 0, 100, 0)
        slider_size = st.slider("Min Size Index", 0, 100, 0)
        n_results = st.slider("Number of Results", 5, 50, 15)
        required_tags = st.multiselect(
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

    try:
        from src.search.autocomplete import suggest_rich  # noqa: E402
        suggestions = suggest_rich(query, limit=25)
    except Exception:
        suggestions = []
    if suggestions:
        picked = st.selectbox("Suggestions", ["(keep typing)"] + suggestions, index=0)
        if picked != "(keep typing)":
            query = picked

    if query:
        # --- QUERY INTENTS (coach-speak -> filters) ---
        try:
            from src.search.coach_dictionary import infer_intents_verbose, INTENTS  # noqa: E402
            intents = infer_intents_verbose(query)
        except Exception:
            intents = {}
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

            intent_dog = max(intent_dog, int(intent.traits.get("dog", 0) * w))
            intent_menace = max(intent_menace, int(intent.traits.get("menace", 0) * w))
            intent_unselfish = max(intent_unselfish, int(intent.traits.get("unselfish", 0) * w))
            intent_tough = max(intent_tough, int(intent.traits.get("tough", 0) * w))
            intent_rim = max(intent_rim, int(intent.traits.get("rim", 0) * w))
            intent_shot = max(intent_shot, int(intent.traits.get("shot", 0) * w))
            # size/measurables intent triggers
            if intent is INTENTS.get("size_measurables"):
                intent_size = max(intent_size, 70)
            # make intent tags soft (don't hard-filter)
            intent_tags = list(set(intent_tags + list(intent.tags)))
            exclude_tags |= intent.exclude_tags

        # Role hints to lightly nudge tags
        if "guard" in role_hints:
            intent_tags = list(set(intent_tags + ["drive", "pnr"]))
        if "wing" in role_hints:
            intent_tags = list(set(intent_tags + ["3pt", "deflection"]))
        if "big" in role_hints:
            intent_tags = list(set(intent_tags + ["rim_pressure", "block", "post_up"]))

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

            # Trait averages (for strengths/weaknesses)
            cur.execute(
                """
                SELECT AVG(dog_index), AVG(menace_index), AVG(unselfish_index),
                       AVG(toughness_index), AVG(rim_pressure_index), AVG(shot_making_index), AVG(size_index)
                FROM player_traits
                """
            )
            avg_row = cur.fetchone() or (0, 0, 0, 0, 0, 0, 0)
            trait_avg = {
                "dog": avg_row[0] or 0,
                "menace": avg_row[1] or 0,
                "unselfish": avg_row[2] or 0,
                "tough": avg_row[3] or 0,
                "rim": avg_row[4] or 0,
                "shot": avg_row[5] or 0,
                "size": avg_row[6] or 0,
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

                if dog_index is not None and dog_index < slider_dog:
                    continue
                if menace_index is not None and menace_index < slider_menace:
                    continue
                if unselfish_index is not None and unselfish_index < slider_unselfish:
                    continue
                if tough_index is not None and tough_index < slider_tough:
                    continue
                if rim_index is not None and rim_index < slider_rim:
                    continue
                if shot_index is not None and shot_index < slider_shot:
                    continue
                if t.get("size") is not None and t.get("size") < slider_size:
                    continue

                play_tags = tag_play(desc)
                if "non_possession" in play_tags:
                    continue
                if exclude_tags and set(play_tags).intersection(exclude_tags):
                    continue
                # Only hard-filter by user-selected Required Tags
                if required_tags and not set(required_tags).issubset(set(play_tags)):
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

                # soft intent boosts (do not hard-filter)
                score += max(0, (t.get("dog") or 0) - intent_dog) * 0.1
                score += max(0, (t.get("menace") or 0) - intent_menace) * 0.1
                score += max(0, (t.get("unselfish") or 0) - intent_unselfish) * 0.1
                score += max(0, (t.get("tough") or 0) - intent_tough) * 0.1
                score += max(0, (t.get("rim") or 0) - intent_rim) * 0.1
                score += max(0, (t.get("shot") or 0) - intent_shot) * 0.1

                score += len(set(play_tags).intersection(set(intent_tags))) * 10

                if "turnover" in play_tags:
                    score -= 8

                home, away, video = matchups.get(gid, ("Unknown", "Unknown", None))

                # Strengths/weaknesses vs averages
                trait_map = {
                    "dog": ("Dog", dog_index),
                    "menace": ("Menace", menace_index),
                    "unselfish": ("Unselfish", unselfish_index),
                    "tough": ("Toughness", tough_index),
                    "rim": ("Rim Pressure", rim_index),
                    "shot": ("Shot Making", shot_index),
                    "size": ("Size", t.get("size")),
                }
                strengths = []
                weaknesses = []
                for key, (label, val) in trait_map.items():
                    if val is None:
                        continue
                    avg = trait_avg.get(key, 0)
                    if val >= avg + 10:
                        strengths.append(label)
                    elif val <= avg - 10:
                        weaknesses.append(label)
                strengths = strengths[:2]
                weaknesses = weaknesses[:2]

                # Human-readable "why"
                reason_parts = []
                if intent_unselfish and (unselfish_index or 0) >= intent_unselfish:
                    reason_parts.append("high unselfishness")
                if intent_tough and (tough_index or 0) >= intent_tough:
                    reason_parts.append("tough/competitive")
                if intent_dog and (dog_index or 0) >= intent_dog:
                    reason_parts.append("dog mentality")
                if intent_menace and (menace_index or 0) >= intent_menace:
                    reason_parts.append("defensive menace")
                if intent_rim and (rim_index or 0) >= intent_rim:
                    reason_parts.append("rim pressure")
                if intent_shot and (shot_index or 0) >= intent_shot:
                    reason_parts.append("shot making")
                if not reason_parts and strengths:
                    reason_parts = [s.lower() for s in strengths]
                reason = " — ".join(reason_parts) if reason_parts else "solid all-around fit"

                # External profile links (search)
                team = home if player_name and player_name in desc else home
                q = f"{player_name or 'player'} {home} basketball"
                espn_url = f"https://www.espn.com/search/_/q/{q.replace(' ', '%20')}"
                ncaa_url = f"https://stats.ncaa.org/search/m?search={q.replace(' ', '+')}"

                rows.append({
                    "Matchup": f"{home} vs {away}",
                    "Clock": clock,
                    "Player": (player_name or "Unknown"),
                    "Why": reason,
                    "Strengths": ", ".join(strengths) if strengths else "—",
                    "Weaknesses": ", ".join(weaknesses) if weaknesses else "—",
                    "Profile": f"[ESPN]({espn_url}) | [NCAA]({ncaa_url})",
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

                # Top 5 player cards
                st.markdown("### Top 5 Players")
                top_players = []
                for player, clips in grouped.items():
                    avg_score = sum(c.get("Score", 0) for c in clips[:3]) / max(1, len(clips[:3]))
                    top_players.append((player, avg_score, clips[0]))
                top_players.sort(key=lambda x: x[1], reverse=True)

                cols = st.columns(5)
                for idx, (player, score, clip) in enumerate(top_players[:5]):
                    with cols[idx]:
                        st.markdown(f"**{player}**")
                        st.caption(f"Score: {score:.1f}")
                        st.caption(clip.get("Matchup", ""))

                st.markdown("### Results")

                for player, clips in grouped.items():
                    st.markdown(f"## {player}")
                    for r in clips[:3]:
                        with st.container():
                            st.markdown(f"**{r['Matchup']}** @ {r['Clock']}")
                            st.caption(f"Tags: {r['Tags']} | Score: {r.get('Score', 0)}")
                            st.write(r["Play"])
                            st.caption(f"Why: {r.get('Why','')}")
                            st.caption(f"Strengths: {r.get('Strengths','—')} | Weaknesses: {r.get('Weaknesses','—')}")
                            st.markdown(r.get("Profile", ""), unsafe_allow_html=True)
                            if r.get("Video") and r["Video"] != "-":
                                try:
                                    st.video(r["Video"])
                                except Exception:
                                    pass
                            st.divider()
            else:
                st.info("No results after filters.")
