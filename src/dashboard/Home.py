import sys
import streamlit as st
from pathlib import Path
import zipfile
import json
import math
import re
import os
from difflib import SequenceMatcher
import requests

# --- 1. SETUP PATHS ---
# Ensure repo root is on sys.path so imports work
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DB_PATH = REPO_ROOT / "data" / "skout.db"

# --- 2. PAGE CONFIGURATION ---
WORDMARK_DARK_URL = "https://portalrecruit.github.io/PortalRecruit/PORTALRECRUIT_LOGO_BANNER_V3.jpg"
from src.dashboard.theme import inject_background

st.set_page_config(
    page_title="PortalRecruit | Search",
    layout="wide",
    page_icon="https://portalrecruit.github.io/PortalRecruit/PR_LOGO_BBALL_SQUARE_HARDWOODBG_2.PNG",
    initial_sidebar_state="expanded", # Changed to expanded for navigation
)
inject_background()

# --- 3. HELPER FUNCTIONS ---
def _restore_vector_db_if_needed() -> bool:
    """Rebuild vector_db from split zip parts if missing."""
    db_path = REPO_ROOT / "data" / "vector_db" / "chroma.sqlite3"
    if db_path.exists():
        return True

    parts = sorted((REPO_ROOT / "data").glob("vector_db.zip.part*"))
    if not parts:
        return False

    zip_path = REPO_ROOT / "data" / "vector_db.zip"
    if not zip_path.exists():
        with open(zip_path, "wb") as out:
            for part in parts:
                out.write(part.read_bytes())

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(REPO_ROOT / "data")
    except Exception:
        return False

    return db_path.exists()


@st.cache_data(show_spinner=False)
def _load_players_index():
    try:
        import sqlite3
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT player_id, full_name, position, team_id, class_year FROM players")
        rows = cur.fetchall()
        con.close()
        players = []
        for r in rows:
            players.append({
                "player_id": r[0],
                "full_name": r[1] or "",
                "position": r[2] or "",
                "team_id": r[3] or "",
                "class_year": r[4] or "",
            })
        return players
    except Exception:
        return []


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _looks_like_name(query: str) -> bool:
    q = (query or "").strip()
    if len(q) < 3:
        return False
    if any(ch.isdigit() for ch in q):
        return False
    parts = q.split()
    return 1 <= len(parts) <= 3


def _resolve_name_query(query: str):
    if not query or not _looks_like_name(query):
        return {"mode": "none", "matches": []}
    players = _load_players_index()
    if not players:
        return {"mode": "none", "matches": []}
    norm_q = _norm_name(query)
    exact = [p for p in players if _norm_name(p["full_name"]) == norm_q]
    if exact:
        return {"mode": "exact_single" if len(exact) == 1 else "exact_multi", "matches": exact}

    # fuzzy match
    scored = []
    for p in players:
        name_norm = _norm_name(p["full_name"])
        if not name_norm:
            continue
        score = SequenceMatcher(None, norm_q, name_norm).ratio()
        scored.append((score, p))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [p for _, p in scored[:5]]
    if not scored:
        return {"mode": "none", "matches": []}
    best = scored[0][0]
    if best >= 0.90:
        return {"mode": "fuzzy_multi", "matches": top}
    return {"mode": "none", "matches": []}



@st.cache_data(show_spinner=False)
def _lookup_player_id_by_name(name: str):
    if not name:
        return None
    try:
        import sqlite3
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT player_id FROM players WHERE full_name = ? LIMIT 1", (name,))
        row = cur.fetchone()
        con.close()
        return row[0] if row else None
    except Exception:
        return None

def _get_player_profile(player_id: str):
    import sqlite3
    profile = {}
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT player_id, full_name, position, team_id, class_year, height_in, weight_lb FROM players WHERE player_id=?", (player_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        return None
    profile.update({
        "player_id": row[0],
        "name": row[1],
        "position": row[2],
        "team_id": row[3],
        "class_year": row[4],
        "height_in": row[5],
        "weight_lb": row[6],
    })
    # traits
    cur.execute("SELECT * FROM player_traits WHERE player_id=?", (player_id,))
    trow = cur.fetchone()
    traits = {}
    if trow:
        cols = [d[0] for d in cur.description]
        traits = dict(zip(cols, trow))
    profile["traits"] = traits

    # season stats
    cur.execute("SELECT gp, possessions, points, fg_percent, shot3_percent, ft_percent, turnover FROM player_season_stats WHERE player_id=?", (player_id,))
    srow = cur.fetchone()
    if srow:
        profile["stats"] = {
            "gp": srow[0],
            "possessions": srow[1],
            "points": srow[2],
            "fg_percent": srow[3],
            "shot3_percent": srow[4],
            "ft_percent": srow[5],
            "turnover": srow[6],
        }
    else:
        profile["stats"] = {}

    # plays
    cur.execute("SELECT play_id, description, game_id, clock_display FROM plays WHERE player_id=? LIMIT 25", (player_id,))
    plays = cur.fetchall()
    profile["plays"] = plays

    # matchups
    game_ids = list({p[2] for p in plays})
    matchups = {}
    if game_ids:
        ph = ",".join(["?"] * len(game_ids))
        cur.execute(f"SELECT game_id, home_team, away_team FROM games WHERE game_id IN ({ph})", game_ids)
        matchups = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    profile["matchups"] = matchups

    con.close()
    return profile


@st.cache_data(show_spinner=False)
def _llm_scout_breakdown(profile: dict) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _scout_breakdown(profile)

    name = profile.get("name", "Player")
    traits = profile.get("traits", {}) or {}
    stats = profile.get("stats", {}) or {}
    plays = profile.get("plays", [])[:5]
    clips = []
    for play_id, desc, game_id, clock in plays:
        if desc:
            clips.append({"id": play_id, "desc": desc})

    prompt = f"""
You are a veteran college basketball recruiter. Write a concise, human, scout-style profile for {name}.
Use the traits and stats below. Speak in coach-speak. Include 3–5 short citations that reference the clips list
using the format [clip:ID].

Traits: {traits}
Stats: {stats}
Clips: {clips}

Write 1 short paragraph and end with a 1-sentence summary of fit.
""".strip()

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a veteran college basketball recruiter."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 260,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return _scout_breakdown(profile)

def _scout_breakdown(profile: dict) -> str:
    name = profile.get("name", "Player")
    traits = profile.get("traits", {}) or {}
    strengths = []
    weaknesses = []
    # simple trait extraction
    trait_map = [
        ("dog_index", "dog mentality"),
        ("menace_index", "defensive menace"),
        ("unselfish_index", "unselfish playmaking"),
        ("toughness_index", "tough, competitive edge"),
        ("rim_pressure_index", "rim pressure"),
        ("shot_making_index", "shot making"),
        ("size_index", "size/length"),
    ]
    for key, label in trait_map:
        val = traits.get(key)
        if val is None:
            continue
        if val >= 70:
            strengths.append(label)
        elif val <= 35:
            weaknesses.append(label)

    lines = []
    if strengths:
        lines.append(f"{name} consistently shows **{', '.join(strengths[:3])}** on film and in the data.")
    if weaknesses:
        lines.append(f"Primary growth areas: **{', '.join(weaknesses[:2])}**.")

    # cite plays
    plays = profile.get("plays", [])[:3]
    if plays:
        cites = []
        for p in plays:
            pid = p[0]
            desc = p[1]
            if desc:
                cites.append(f"[{desc[:80]}...] (#clip-{pid})")
        if cites:
            lines.append("Example clips: " + " ".join(cites))

    if not lines:
        lines.append(f"{name} profiles as a balanced contributor with a mix of skill and competitive traits.")
    return " ".join(lines)


def _render_profile_overlay(player_id: str):
    profile = _get_player_profile(player_id)
    if not profile:
        st.warning("Player not found.")
        return
    title = profile.get("name", "Player Profile")

    def body():
        st.markdown(f"## {title}")
        meta = []
        if profile.get("position"): meta.append(profile["position"])
        if profile.get("height_in") and profile.get("weight_lb"):
            meta.append(f"{profile['height_in']}in / {profile['weight_lb']}lb")
        if profile.get("class_year"): meta.append(f"Class: {profile['class_year']}")
        if profile.get("team_id"): meta.append(f"Team: {profile['team_id']}")
        if meta:
            st.caption(" • ".join(meta))

        st.markdown("### Scout Breakdown")
        breakdown = _llm_scout_breakdown(profile)
        breakdown = re.sub(r"\[clip:(\d+)\]", r"[clip](#clip-\\1)", breakdown)
        st.write(breakdown)

        # traits + strengths/weaknesses
        traits = profile.get("traits", {}) or {}
        if traits:
            st.markdown("### Traits")
            strengths = []
            weaknesses = []
            for key, label in [
                ("dog_index", "Dog"),
                ("menace_index", "Menace"),
                ("unselfish_index", "Unselfish"),
                ("toughness_index", "Toughness"),
                ("rim_pressure_index", "Rim Pressure"),
                ("shot_making_index", "Shot Making"),
                ("gravity_index", "Gravity"),
                ("size_index", "Size"),
            ]:
                val = traits.get(key)
                if val is None:
                    continue
                if val >= 70:
                    strengths.append(label)
                elif val <= 35:
                    weaknesses.append(label)
                st.progress(min(100, max(0, int(val))))
                st.caption(f"{label}: {val:.1f}" if isinstance(val, (int, float)) else f"{label}: {val}")
            if strengths:
                st.markdown(f"**Strengths:** {', '.join(strengths[:4])}")
            if weaknesses:
                st.markdown(f"**Weaknesses:** {', '.join(weaknesses[:3])}")

        # clips (anchors for citations)
        plays = profile.get("plays", [])
        matchups = profile.get("matchups", {})
        if plays:
            st.markdown("### Clips")
            for play_id, desc, game_id, clock in plays[:15]:
                st.markdown(f"<a name='clip-{play_id}'></a>", unsafe_allow_html=True)
                home, away = matchups.get(game_id, ("Unknown", "Unknown"))
                st.markdown(f"**{home} vs {away}** @ {clock}")
                st.write(desc)
                st.divider()

    if hasattr(st, "dialog"):
        with st.dialog("Player Profile"):
            if st.button("✕ Close", key="close_profile_top"):
                st.query_params.pop("player", None)
                st.rerun()
            body()
    else:
        st.markdown("---")
        if st.button("✕ Close Profile", key="close_profile"):
            st.query_params.pop("player", None)
            st.rerun()
        body()

def check_ingestion_status():
    """
    Checks if the vector database exists.
    """
    _restore_vector_db_if_needed()
    db_path = REPO_ROOT / "data" / "vector_db" / "chroma.sqlite3"
    return db_path.exists()

def render_header():
    st.markdown(
        f"""
        <div class="pr-hero">
          <img src="{WORDMARK_DARK_URL}" style="max-width:560px; width:min(560px, 92vw); height:auto; object-fit:contain;" />
          <div class="pr-hero-sub" style="font-family: var(--pr-font-body);">
            <strong>How to use:</strong> Describe the player you need in coach‑speak (e.g., “downhill guard who can guard”). Use filters to narrow.
            <br/>
            <strong>How to read results:</strong> Ranked by semantic fit + traits. Use <em>Why</em>, <em>Strengths</em>, and tags to interpret.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def _zscore(val, mean, std):
    if std == 0:
        return 0.0
    return (val - mean) / std


def _load_nba_archetypes():
    path = REPO_ROOT / "data" / "nba_archetypes.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []

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

    # Route to player profile page via query params
    if "player" in st.query_params and st.query_params["player"]:
        pid = st.query_params["player"]
        if _get_player_profile(pid):
            _render_profile_overlay(pid)
            st.stop()
        else:
            st.query_params.pop("player", None)
            st.warning("Player not found.")
    
    # This is where your Search UI lives. 
    # Ideally, put this in a separate file like `src/dashboard/search_ui.py` and import it.
    # For now, I'll place the logic block here.
    
    st.markdown("### 🔍 Semantic Player Search")

    # Advanced Filters (collapsed by default)
    min_dog = min_menace = min_unselfish = min_tough = min_rim = min_shot = min_gravity = 0
    min_size = 0
    slider_dog = slider_menace = slider_unselfish = slider_tough = slider_rim = slider_shot = slider_gravity = 0
    slider_size = 0
    intent_dog = intent_menace = intent_unselfish = intent_tough = intent_rim = intent_shot = intent_gravity = 0
    intent_size = 0
    n_results = 15
    tag_filter = []
    intent_tags = []
    required_tags = []
    finishing_intent = False

    with st.sidebar.expander("Advanced Filters", expanded=False):
        slider_dog = st.slider("Min Dog Index", 0, 100, 0)
        slider_menace = st.slider("Min Defensive Menace", 0, 100, 0)
        slider_unselfish = st.slider("Min Unselfishness", 0, 100, 0)
        slider_tough = st.slider("Min Toughness", 0, 100, 0)
        slider_rim = st.slider("Min Rim Pressure", 0, 100, 0)
        slider_shot = st.slider("Min Shot Making", 0, 100, 0)
        slider_gravity = st.slider("Min Gravity Well", 0, 100, 0)
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

    # Name-aware search routing
    name_resolution = _resolve_name_query(query)
    if name_resolution.get("mode") == "exact_single":
        st.query_params["player"] = name_resolution["matches"][0]["player_id"]
        st.rerun()
    elif name_resolution.get("mode") in {"exact_multi", "fuzzy_multi"}:
        st.markdown("### Did you mean")
        cols = st.columns(2)
        for i, p in enumerate(name_resolution["matches"]):
            with cols[i % 2]:
                st.markdown(f"**{p['full_name']}**")
                st.caption(f"{p.get('position','')} | Team {p.get('team_id','')} | {p.get('class_year','')}")
                if st.button("View Profile", key=f"didyoumean_{p['player_id']}"):
                    st.session_state.profile_player_id = p["player_id"]
                    _render_profile_overlay(st.session_state.profile_player_id)
                    st.stop()
        st.stop()

    if query:
        # --- QUERY INTENTS (coach-speak -> filters) ---
        try:
            from src.search.coach_dictionary import infer_intents_verbose, INTENTS  # noqa: E402
            # Simple logic parsing (and/or/but) for better intent handling
            q_lower = (query or "").lower()
            if " or " in q_lower:
                logic = "or"
                parts = [p.strip() for p in q_lower.split(" or ") if p.strip()]
            elif " and " in q_lower or " but " in q_lower:
                logic = "and"
                parts = [p.strip() for p in q_lower.replace(" but ", " and ").split(" and ") if p.strip()]
            else:
                logic = "single"
                parts = [q_lower]

            # Numeric filters (over/under/above/below)
            import re
            numeric_filters = []
            pattern = re.compile(r"\b(over|under|above|below|at least|atleast|at most|atmost|more than|less than)\s+(\d+(?:\.\d+)?)%?\s*(3pt|3pt%|3pt\s*%|three|three point|three-point|ft|free throw|free-throw|fg|field goal|field-goal|shot)\b")
            for m in pattern.finditer(q_lower):
                comp = m.group(1)
                val = float(m.group(2))
                stat = m.group(3)
                numeric_filters.append((comp, val, stat))

            intents = {}
            for part in parts:
                intents.update(infer_intents_verbose(part))
        except Exception:
            intents = {}
        exclude_tags = set()
        role_hints = set()
        explain = []
        matched_phrases = []
        leadership_intent = "leadership" in intents
        resilience_intent = "resilience" in intents
        defensive_big_intent = "defensive_big" in intents
        clutch_intent = "clutch" in intents
        undervalued_intent = "undervalued" in intents
        finishing_intent = "finishing" in intents

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
            intent_gravity = max(intent_gravity, int(intent.traits.get("gravity", 0) * w))
            # size/measurables intent triggers
            if intent is INTENTS.get("size_measurables"):
                intent_size = max(intent_size, 70)
            # make intent tags soft (don't hard-filter)
            intent_tags = list(set(intent_tags + list(intent.tags)))
            exclude_tags |= intent.exclude_tags

            # If "and" logic, treat intent tags as required
            if logic == "and":
                required_tags = list(set(required_tags + list(intent.tags)))

        # Role hints to lightly nudge tags
        if "guard" in role_hints:
            intent_tags = list(set(intent_tags + ["drive", "pnr"]))
        if "wing" in role_hints:
            intent_tags = list(set(intent_tags + ["3pt", "deflection"]))
        if "big" in role_hints:
            intent_tags = list(set(intent_tags + ["rim_pressure", "block", "post_up"]))

        # Finishing intent should hard-require rim finish + made
        if finishing_intent:
            required_tags = list(set(required_tags + ["rim_finish", "layup", "dunk", "made"]))

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
            # try restore once more if parts exist
            _restore_vector_db_if_needed()
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

        # Optional cross-encoder re-rank for better semantic precision
        try:
            from sentence_transformers import CrossEncoder
            docs = results.get("documents", [[]])[0]
            ids = results.get("ids", [[]])[0]
            if docs and ids:
                cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                pairs = [[expanded_query, d] for d in docs]
                scores = cross.predict(pairs)
                reranked = sorted(zip(ids, scores, docs), key=lambda x: x[1], reverse=True)
                play_ids = [r[0] for r in reranked]
            else:
                play_ids = results.get("ids", [[]])[0]
        except Exception:
            play_ids = results.get("ids", [[]])[0]
        if not play_ids:
            st.warning("No results found.")
        else:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()

            # Ensure trait columns exist for older DBs
            try:
                cur.execute("PRAGMA table_info(player_traits)")
                existing_cols = {r[1] for r in cur.fetchall()}
                needed = {
                    "leadership_index",
                    "resilience_index",
                    "defensive_big_index",
                    "clutch_index",
                    "undervalued_index",
                    "gravity_index",
                }
                for col in needed:
                    if col not in existing_cols:
                        cur.execute(f"ALTER TABLE player_traits ADD COLUMN {col} REAL")
                conn.commit()
            except Exception:
                pass

            # Pull plays
            try:
                cur.execute("SELECT 1 FROM plays LIMIT 1")
            except Exception:
                st.error("Database missing plays table. Ensure skout.db is available on the host.")
                conn.close()
                st.stop()

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
                           toughness_index, rim_pressure_index, shot_making_index, gravity_index, size_index,
                           leadership_index, resilience_index, defensive_big_index, clutch_index,
                           undervalued_index
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
                        "gravity": r[7],
                        "size": r[8],
                        "leadership": r[9],
                        "resilience": r[10],
                        "defensive_big": r[11],
                        "clutch": r[12],
                        "undervalued": r[13],
                    }
                    for r in cur.fetchall()
                }

            # Trait averages (for strengths/weaknesses)
            cur.execute(
                """
                SELECT AVG(dog_index), AVG(menace_index), AVG(unselfish_index),
                       AVG(toughness_index), AVG(rim_pressure_index), AVG(shot_making_index), AVG(gravity_index),
                       AVG(size_index), AVG(leadership_index), AVG(resilience_index), AVG(defensive_big_index),
                       AVG(clutch_index), AVG(undervalued_index)
                FROM player_traits
                """
            )
            avg_row = cur.fetchone() or (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            trait_avg = {
                "dog": avg_row[0] or 0,
                "menace": avg_row[1] or 0,
                "unselfish": avg_row[2] or 0,
                "tough": avg_row[3] or 0,
                "rim": avg_row[4] or 0,
                "shot": avg_row[5] or 0,
                "gravity": avg_row[6] or 0,
                "size": avg_row[7] or 0,
                "leadership": avg_row[8] or 0,
                "resilience": avg_row[9] or 0,
                "defensive_big": avg_row[10] or 0,
                "clutch": avg_row[11] or 0,
                "undervalued": avg_row[12] or 0,
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

            # Load player positions for role filtering
            player_positions = {}
            try:
                conn_pos = sqlite3.connect(DB_PATH)
                cur_pos = conn_pos.cursor()
                cur_pos.execute("SELECT player_id, position FROM players")
                player_positions = {r[0]: (r[1] or "") for r in cur_pos.fetchall()}
                conn_pos.close()
            except Exception:
                player_positions = {}

            # Load player meta (name, team, height/weight)
            player_meta = {}
            try:
                conn_meta = sqlite3.connect(DB_PATH)
                cur_meta = conn_meta.cursor()
                cur_meta.execute("SELECT player_id, full_name, position, team_id, height_in, weight_lb FROM players")
                for r in cur_meta.fetchall():
                    player_meta[r[0]] = {
                        "full_name": r[1] or "",
                        "position": r[2] or "",
                        "team_id": r[3] or "",
                        "height_in": r[4],
                        "weight_lb": r[5],
                    }
                conn_meta.close()
            except Exception:
                player_meta = {}

            # Preload season stats + player names + full traits for similarity
            player_stats = {}
            player_names = {}
            traits_all = {}
            try:
                conn2 = sqlite3.connect(DB_PATH)
                cur2 = conn2.cursor()
                cur2.execute(
                    """
                    SELECT player_id, team_id, gp, possessions, points,
                           fg_percent, shot2_percent, shot3_percent, ft_percent,
                           fg_attempt, shot2_attempt, shot3_attempt, turnover
                    FROM player_season_stats
                    """
                )
                for r in cur2.fetchall():
                    player_stats[r[0]] = {
                        "team_id": r[1],
                        "gp": r[2],
                        "possessions": r[3],
                        "points": r[4],
                        "fg_percent": r[5],
                        "shot2_percent": r[6],
                        "shot3_percent": r[7],
                        "ft_percent": r[8],
                        "fg_attempt": r[9],
                        "shot2_attempt": r[10],
                        "shot3_attempt": r[11],
                        "turnover": r[12],
                    }
                cur2.execute("SELECT player_id, full_name FROM players")
                for r in cur2.fetchall():
                    player_names[r[0]] = r[1]
                cur2.execute(
                    """
                    SELECT player_id, dog_index, menace_index, unselfish_index,
                           toughness_index, rim_pressure_index, shot_making_index, gravity_index, size_index,
                           leadership_index, resilience_index, defensive_big_index, clutch_index, undervalued_index
                    FROM player_traits
                    """
                )
                for r in cur2.fetchall():
                    traits_all[r[0]] = {
                        "dog": r[1],
                        "menace": r[2],
                        "unselfish": r[3],
                        "tough": r[4],
                        "rim": r[5],
                        "shot": r[6],
                        "gravity": r[7],
                        "size": r[8],
                        "leadership": r[9],
                        "resilience": r[10],
                        "defensive_big": r[11],
                        "clutch": r[12],
                        "undervalued": r[13],
                    }
                conn2.close()
            except Exception:
                player_stats = {}
                player_names = {}
                traits_all = {}

            # Build global means/stds for normalization
            def _collect_vals(key):
                vals = [v.get(key) for v in player_stats.values() if v.get(key) is not None]
                return vals

            stat_keys = [
                "points", "possessions", "fg_percent", "shot3_percent", "ft_percent",
                "fg_attempt", "shot2_attempt", "shot3_attempt", "turnover",
            ]
            stat_means = {k: (_safe_float(sum(_collect_vals(k))) / max(1, len(_collect_vals(k)))) for k in stat_keys}
            stat_stds = {
                k: math.sqrt(sum(((_safe_float(v) - stat_means[k]) ** 2) for v in _collect_vals(k)) / max(1, len(_collect_vals(k))))
                for k in stat_keys
            }

            rows = []
            for pid, desc, gid, clock, player_id, player_name in play_rows:
                t = traits.get(player_id, {})
                dog_index = t.get("dog")
                menace_index = t.get("menace")
                unselfish_index = t.get("unselfish")
                tough_index = t.get("tough")
                rim_index = t.get("rim")
                shot_index = t.get("shot")
                gravity_index = t.get("gravity")

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
                if gravity_index is not None and gravity_index < slider_gravity:
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

                # Numeric filters (e.g., over 36% from 3)
                if numeric_filters:
                    pstats = player_stats.get(player_id, {})
                    allow = True
                    for comp, val, stat in numeric_filters:
                        stat_val = None
                        if "3" in stat or "three" in stat:
                            stat_val = _safe_float(pstats.get("shot3_percent")) * 100
                        elif stat in {"ft", "free throw", "free-throw"}:
                            stat_val = _safe_float(pstats.get("ft_percent")) * 100
                        elif stat in {"fg", "field goal", "field-goal", "shot"}:
                            stat_val = _safe_float(pstats.get("fg_percent")) * 100

                        if stat_val is None:
                            continue
                        if comp in {"over", "above", "more than", "at least", "atleast"} and not (stat_val >= val):
                            allow = False
                        if comp in {"under", "below", "less than", "at most", "atmost"} and not (stat_val <= val):
                            allow = False
                    if not allow:
                        continue

                # Role-based position filtering
                pos = (player_positions.get(player_id) or "").upper()
                if "guard" in role_hints and not ("G" in pos or "PG" in pos or "SG" in pos):
                    continue
                if "wing" in role_hints and not ("F" in pos or "W" in pos or "G/F" in pos or "F/G" in pos or "SF" in pos):
                    continue
                if "big" in role_hints and not ("C" in pos or "F/C" in pos or "PF" in pos):
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
                    ("gravity", 0.6),
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
                score += max(0, (t.get("gravity") or 0) - intent_gravity) * 0.1

                score += len(set(play_tags).intersection(set(intent_tags))) * 10

                if leadership_intent and t.get("leadership"):
                    score += t.get("leadership") * 15

                if resilience_intent and t.get("resilience"):
                    score += t.get("resilience") * 12

                if defensive_big_intent and t.get("defensive_big"):
                    score += t.get("defensive_big") * 18

                if clutch_intent and t.get("clutch"):
                    score += t.get("clutch") * 15

                if undervalued_intent and t.get("undervalued"):
                    score += t.get("undervalued") * 14

                if "turnover" in play_tags:
                    score -= 8

                # Evidence-first ranking: made plays + primary action
                if "made" in play_tags:
                    score += 12
                if "missed" in play_tags:
                    score -= 6
                if player_name and player_name in (desc or ""):
                    score += 6

                home, away, video = matchups.get(gid, ("Unknown", "Unknown", None))

                # Strengths/weaknesses vs averages
                trait_map = {
                    "dog": ("Dog", dog_index),
                    "menace": ("Menace", menace_index),
                    "unselfish": ("Unselfish", unselfish_index),
                    "tough": ("Toughness", tough_index),
                    "rim": ("Rim Pressure", rim_index),
                    "shot": ("Shot Making", shot_index),
                    "gravity": ("Gravity Well", gravity_index),
                    "size": ("Size", t.get("size")),
                    "leadership": ("Leadership", t.get("leadership")),
                    "resilience": ("Resilience", t.get("resilience")),
                    "defensive_big": ("Defensive Big", t.get("defensive_big")),
                    "clutch": ("Clutch", t.get("clutch")),
                    "undervalued": ("Undervalued", t.get("undervalued")),
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
                if intent_gravity and (gravity_index or 0) >= intent_gravity:
                    reason_parts.append("gravity well")
                if not reason_parts and strengths:
                    reason_parts = [s.lower() for s in strengths]
                reason = " — ".join(reason_parts) if reason_parts else "solid all-around fit"

                # Similar NCAA + NBA comps
                sim_players = []
                nba_floor = nba_ceiling = "—"
                try:
                    # Build player vector
                    pstats = player_stats.get(player_id, {})
                    height_in = _safe_float((t.get("size") or 0))
                    weight_lb = _safe_float(0)
                    try:
                        # pull from players table if available in traits map (size only stored)
                        pass
                    except Exception:
                        pass

                    trait_vec = {
                        "dog": _safe_float(dog_index),
                        "menace": _safe_float(menace_index),
                        "unselfish": _safe_float(unselfish_index),
                        "tough": _safe_float(tough_index),
                        "rim": _safe_float(rim_index),
                        "shot": _safe_float(shot_index),
                        "gravity": _safe_float(gravity_index),
                        "leadership": _safe_float(t.get("leadership")),
                        "resilience": _safe_float(t.get("resilience")),
                        "defensive_big": _safe_float(t.get("defensive_big")),
                        "clutch": _safe_float(t.get("clutch")),
                        "undervalued": _safe_float(t.get("undervalued")),
                    }

                    def _sim_score(pid2):
                        if pid2 == player_id:
                            return -999
                        ps2 = player_stats.get(pid2, {})
                        # exclude same team
                        if pstats.get("team_id") and ps2.get("team_id") == pstats.get("team_id"):
                            return -999
                        t2 = traits_all.get(pid2, {})

                        # physical
                        phys_dist = abs(_safe_float(t.get("size")) - _safe_float(t2.get("size"))) / 100.0

                        # production
                        prod_dist = 0.0
                        for k in stat_keys:
                            prod_dist += abs(
                                _zscore(_safe_float(pstats.get(k)), stat_means[k], stat_stds[k])
                                - _zscore(_safe_float(ps2.get(k)), stat_means[k], stat_stds[k])
                            )
                        prod_dist /= max(1, len(stat_keys))

                        # traits
                        trait_keys = ["dog", "menace", "unselfish", "tough", "rim", "shot", "gravity", "leadership", "resilience", "defensive_big", "clutch", "undervalued"]
                        trait_dist = 0.0
                        for k in trait_keys:
                            trait_dist += abs(_safe_float(t.get(k)) - _safe_float(t2.get(k))) / 100.0
                        trait_dist /= max(1, len(trait_keys))

                        score = 1 / (1 + (0.35 * phys_dist + 0.35 * prod_dist + 0.30 * trait_dist))
                        return score

                    if traits_all:
                        scores = sorted(((pid2, _sim_score(pid2)) for pid2 in traits_all.keys()), key=lambda x: x[1], reverse=True)
                        sim_players = [player_names.get(pid2, "") for pid2, s in scores if s > 0][:2]

                    # NBA comps from archetypes
                    archetypes = _load_nba_archetypes()
                    def _archetype_score(a):
                        phys = abs((_safe_float(t.get("size")) - _safe_float(a.get("height_in", 0))) / 100.0)
                        shot3 = abs(_safe_float(pstats.get("shot3_percent")) - _safe_float(a.get("shot3_percent", 0)))
                        trait_dist = 0.0
                        for k, v in (a.get("traits") or {}).items():
                            trait_dist += abs(_safe_float(trait_vec.get(k)) / 100.0 - _safe_float(v) / 100.0)
                        trait_dist /= max(1, len((a.get("traits") or {})))
                        return 1 / (1 + phys + shot3 + trait_dist)

                    if archetypes:
                        floor_candidates = [a for a in archetypes if a.get("tier") == "floor"]
                        ceil_candidates = [a for a in archetypes if a.get("tier") == "ceiling"]
                        if floor_candidates:
                            nba_floor = max(floor_candidates, key=_archetype_score).get("name")
                        if ceil_candidates:
                            nba_ceiling = max(ceil_candidates, key=_archetype_score).get("name")
                except Exception:
                    pass

                # External profile links (search)
                team = home if player_name and player_name in desc else home
                q = f"{player_name or 'player'} {home} basketball"
                espn_url = f"https://www.espn.com/search/_/q/{q.replace(' ', '%20')}"
                ncaa_url = f"https://stats.ncaa.org/search/m?search={q.replace(' ', '+')}"

                rows.append({
                    "Match": f"{home} vs {away}",
                    "Clock": clock,
                    "Player": (player_name or "Unknown"),
                    "Player ID": player_id,
                    "Position": (player_meta.get(player_id, {}).get("position") if "player_meta" in locals() else ""),
                    "Team": (player_meta.get(player_id, {}).get("team_id") if "player_meta" in locals() else ""),
                    "Height": (player_meta.get(player_id, {}).get("height_in") if "player_meta" in locals() else None),
                    "Weight": (player_meta.get(player_id, {}).get("weight_lb") if "player_meta" in locals() else None),
                    "Why": reason,
                    "Strengths": ", ".join(strengths) if strengths else "—",
                    "Weaknesses": ", ".join(weaknesses) if weaknesses else "—",
                    "Similar NCAA": ", ".join([s for s in sim_players if s]) if sim_players else "—",
                    "NBA Floor": nba_floor or "—",
                    "NBA Ceiling": nba_ceiling or "—",
                    "Profile": f"[ESPN]({espn_url}) | [NCAA]({ncaa_url})",
                    "Dog Index": dog_index,
                    "Menace": menace_index,
                    "Unselfish": unselfish_index,
                    "Toughness": tough_index,
                    "Rim Pressure": rim_index,
                    "Shot Making": shot_index,
                    "Gravity Well": gravity_index,
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
                        pid = clip.get("Player ID")
                        pos = clip.get("Position", "")
                        team = clip.get("Team", "")
                        ht = clip.get("Height")
                        wt = clip.get("Weight")
                        size = f"{ht}in/{wt}lb" if ht and wt else ""
                        meta = " | ".join([s for s in [pos, team, size] if s])
                        label = f"{player}\n{meta}\nScore: {score:.1f}"
                        if pid and st.button(label, key=f"top5_{pid}", use_container_width=True):
                            st.session_state.profile_player_id = pid
                            st.rerun()

                st.markdown("### Results")

                for player, clips in grouped.items():
                    pid = clips[0].get("Player ID") if clips else None
                    pos = clips[0].get("Position", "") if clips else ""
                    team = clips[0].get("Team", "") if clips else ""
                    ht = clips[0].get("Height") if clips else None
                    wt = clips[0].get("Weight") if clips else None
                    size = f"{ht}in/{wt}lb" if ht and wt else ""
                    meta = " | ".join([s for s in [pos, team, size] if s])
                    label = f"{player}\n{meta}\nScore: {clips[0].get('Score',0):.1f}" if clips else player
                    if st.button(label, key=f"player_{pid}", use_container_width=True):
                        if not pid:
                            pid = _lookup_player_id_by_name(player)
                        if pid:
                            st.query_params["player"] = pid
                            st.rerun()
                    # Plays shown in overlay only
            else:
                st.info("No results after filters.")
