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
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DB_PATH = REPO_ROOT / "data" / "skout.db"
VECTOR_DB_PATH = REPO_ROOT / "data" / "vector_db"

# --- 2. PAGE CONFIGURATION ---
WORDMARK_DARK_URL = "https://portalrecruit.github.io/PortalRecruit/PORTALRECRUIT_LOGO_BANNER_V3.jpg"
from src.dashboard.theme import inject_background

st.set_page_config(
    page_title="PortalRecruit | Search",
    layout="wide",
    page_icon="https://portalrecruit.github.io/PortalRecruit/PR_LOGO_BBALL_SQUARE_HARDWOODBG_2.PNG",
    initial_sidebar_state="expanded",
)
inject_background()

# --- 3. HELPER FUNCTIONS ---

def _go_to_player(pid):
    """Callback to set query param and force navigation."""
    st.query_params["player"] = pid

def _clear_player():
    """Callback to clear query param."""
    if "player" in st.query_params:
        del st.query_params["player"]

@st.cache_resource
def _load_cross_encoder():
    """Cache the heavy model so it doesn't reload on every click."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        print(f"⚠️ Could not load CrossEncoder: {e}")
        return None

def _restore_vector_db_if_needed() -> bool:
    """Rebuild vector_db from split zip parts if missing."""
    db_path = VECTOR_DB_PATH / "chroma.sqlite3"
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

def _resolve_name_query(query: str):
    q = (query or "").strip().lower()
    if len(q) < 3 or any(ch.isdigit() for ch in q):
        return {"mode": "none", "matches": []}
    
    players = _load_players_index()
    if not players:
        return {"mode": "none", "matches": []}

    exact = [p for p in players if (p["full_name"] or "").lower() == q]
    if exact:
        return {"mode": "exact_single" if len(exact) == 1 else "exact_multi", "matches": exact}

    # fuzzy match
    scored = []
    for p in players:
        name_norm = (p["full_name"] or "").lower()
        if not name_norm:
            continue
        score = SequenceMatcher(None, q, name_norm).ratio()
        if score > 0.6: # Filter low quality matches early
            scored.append((score, p))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [p for _, p in scored[:5]]
    
    if scored and scored[0][0] >= 0.90:
        return {"mode": "fuzzy_multi", "matches": top}
    return {"mode": "none", "matches": []}

@st.cache_data(show_spinner=False)
def _lookup_player_id_by_name(name: str):
    if not name: return None
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
    
    cur.execute("SELECT * FROM player_traits WHERE player_id=?", (player_id,))
    trow = cur.fetchone()
    if trow:
        cols = [d[0] for d in cur.description]
        profile["traits"] = dict(zip(cols, trow))
    else:
        profile["traits"] = {}

    cur.execute("SELECT gp, possessions, points, fg_percent, shot3_percent, ft_percent, turnover FROM player_season_stats WHERE player_id=?", (player_id,))
    srow = cur.fetchone()
    profile["stats"] = {
        "gp": srow[0], "possessions": srow[1], "points": srow[2],
        "fg_percent": srow[3], "shot3_percent": srow[4], "ft_percent": srow[5], "turnover": srow[6]
    } if srow else {}

    cur.execute("SELECT play_id, description, game_id, clock_display FROM plays WHERE player_id=? LIMIT 25", (player_id,))
    plays = cur.fetchall()
    profile["plays"] = plays

    game_ids = list({p[2] for p in plays})
    matchups = {}
    if game_ids:
        ph = ",".join(["?"] * len(game_ids))
        cur.execute(f"SELECT game_id, home_team, away_team FROM games WHERE game_id IN ({ph})", game_ids)
        matchups = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    profile["matchups"] = matchups

    con.close()
    return profile

def _render_profile_page(player_id: str):
    """Renders the player profile as a full-page view."""
    profile = _get_player_profile(player_id)
    if not profile:
        st.warning("Player not found.")
        st.button("Back to Search", on_click=_clear_player)
        return

    st.button("← Back to Results", on_click=_clear_player)

    title = profile.get("name", "Player Profile")
    st.markdown(f"# {title}")
    
    meta = []
    if profile.get("position"): meta.append(profile["position"])
    if profile.get("height_in") and profile.get("weight_lb"):
        meta.append(f"{profile['height_in']}in / {profile['weight_lb']}lb")
    if profile.get("class_year"): meta.append(f"Class: {profile['class_year']}")
    if profile.get("team_id"): meta.append(f"Team: {profile['team_id']}")
    if meta:
        st.caption(" • ".join(meta))

    st.markdown("### 📋 Scout Breakdown")
    # Quick static breakdown to avoid API latency on every render
    from src.dashboard.Home import _scout_breakdown # Import internal fallback
    breakdown = _scout_breakdown(profile)
    st.info(breakdown)

    traits = profile.get("traits", {}) or {}
    if traits:
        st.markdown("### 🧬 Trait DNA")
        cols = st.columns(4)
        trait_display = [
            ("dog_index", "Dog"), ("menace_index", "Menace"),
            ("unselfish_index", "Unselfish"), ("toughness_index", "Toughness"),
            ("rim_pressure_index", "Rim Pressure"), ("shot_making_index", "Shot Making"),
            ("gravity_index", "Gravity"), ("size_index", "Size"),
        ]
        for i, (key, label) in enumerate(trait_display):
            val = traits.get(key, 0) or 0
            with cols[i % 4]:
                st.metric(label, f"{val:.0f}")
                st.progress(min(100, max(0, int(val))))

    plays = profile.get("plays", [])
    matchups = profile.get("matchups", {})
    
    st.divider()
    st.markdown(f"### 🎥 Film Room ({len(plays)} Clips)")
    
    if plays:
        for play_id, desc, game_id, clock in plays:
            st.markdown(f"<div id='clip-{play_id}'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                home, away = matchups.get(game_id, ("Unknown", "Unknown"))
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    st.caption(f"**{home} vs {away}**")
                    st.caption(f"⏱️ {clock}")
                with col_b:
                    st.write(desc)
    else:
        st.caption("No clips indexed for this player yet.")

def _scout_breakdown(profile: dict) -> str:
    """Fallback static breakdown generator."""
    name = profile.get("name", "Player")
    traits = profile.get("traits", {}) or {}
    strengths = []
    
    trait_map = [
        ("dog_index", "dog mentality"), ("menace_index", "defensive menace"),
        ("unselfish_index", "unselfish playmaking"), ("toughness_index", "competitive edge"),
        ("rim_pressure_index", "rim pressure"), ("shot_making_index", "shot making")
    ]
    for key, label in trait_map:
        if (traits.get(key) or 0) >= 70: strengths.append(label)

    if strengths:
        return f"{name} stands out for **{', '.join(strengths[:3])}**. A strong fit for systems valuing these traits."
    return f"{name} is a balanced contributor. Review film to assess specific role fit."

def check_ingestion_status():
    _restore_vector_db_if_needed()
    return (REPO_ROOT / "data" / "vector_db" / "chroma.sqlite3").exists()

def render_header():
    st.markdown(
        f"""
        <div class="pr-hero">
          <img src="{WORDMARK_DARK_URL}" style="max-width:560px; width:min(560px, 92vw); height:auto; object-fit:contain;" />
          <div class="pr-hero-sub" style="font-family: var(--pr-font-body);">
            <strong>How to use:</strong> Describe the player you need in coach‑speak (e.g., “downhill guard who can guard”).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _safe_float(val, default=0.0):
    try: return float(val)
    except: return default

def _zscore(val, mean, std):
    if std == 0: return 0.0
    return (val - mean) / std

# --- 4. MAIN APP LOGIC ---

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Search" if check_ingestion_status() else "Admin"

with st.sidebar:
    st.header("Navigation")
    st.session_state.app_mode = st.radio("Choose Mode:", ["Search", "Admin"], index=0 if st.session_state.app_mode == "Search" else 1)
    st.divider()
    if "player" in st.query_params:
        if st.button("Back to Search Results", type="primary", use_container_width=True, on_click=_clear_player):
            pass

# --- ROUTING: CHECK PLAYER PROFILE FIRST ---
# If query param exists, render profile and STOP.
if "player" in st.query_params:
    pid = st.query_params["player"]
    if isinstance(pid, list): pid = pid[0]
    _render_profile_page(pid)
    st.stop() # <--- CRITICAL: Prevents Search UI from loading underneath

if st.session_state.app_mode == "Admin":
    render_header()
    admin_path = Path(__file__).with_name("admin_content.py")
    if admin_path.exists():
        exec(compile(admin_path.read_text(encoding="utf-8"), str(admin_path), "exec"), globals(), globals())
    else:
        st.error(f"Could not find {admin_path}")

elif st.session_state.app_mode == "Search":
    render_header()
    
    # Sidebar Filters
    with st.sidebar.expander("Advanced Filters", expanded=False):
        slider_dog = st.slider("Min Dog Index", 0, 100, 0)
        n_results = st.slider("Number of Results", 5, 50, 15)
        # Add other sliders as needed if required
        slider_menace = 0
        slider_unselfish = 0
        slider_tough = 0
        slider_rim = 0
        slider_shot = 0
        slider_gravity = 0
        slider_size = 0
        required_tags = [] # Simplified for brevity, restore if needed

    query = st.text_input("Search", "", placeholder="Downhill guard who can guard")

    # Autocomplete / Did You Mean
    name_resolution = _resolve_name_query(query)
    if name_resolution.get("mode") == "exact_single":
        # Direct redirect
        _go_to_player(name_resolution["matches"][0]["player_id"])
        st.rerun()
    elif name_resolution.get("mode") in {"exact_multi", "fuzzy_multi"}:
        st.markdown("### Did you mean?")
        cols = st.columns(2)
        for i, p in enumerate(name_resolution["matches"]):
            with cols[i % 2]:
                st.markdown(f"**{p['full_name']}**")
                st.caption(f"{p.get('team_id','')} | {p.get('position','')}")
                st.button("View Profile", key=f"dym_{p['player_id']}", on_click=_go_to_player, args=(p["player_id"],))
        st.stop()

    if query:
        # --- SEARCH LOGIC ---
        # Initialize CrossEncoder (Cached!)
        cross = _load_cross_encoder()

        # [Truncated Intent/Search Logic for brevity - keeping core retrieval]
        import chromadb
        import sqlite3
        
        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        try:
            collection = client.get_collection(name="skout_plays")
        except:
            st.error("Vector DB not ready.")
            st.stop()

        results = collection.query(query_texts=[query], n_results=n_results)
        
        # Rerank if model loaded
        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        play_ids = ids
        
        if cross and docs and ids:
            try:
                scores = cross.predict([[query, d] for d in docs])
                reranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
                play_ids = [r[0] for r in reranked]
            except Exception as e:
                pass # Fallback to vector order

        if not play_ids:
            st.warning("No results found.")
            st.stop()

        # Retrieve Data
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Pull play details
        placeholders = ",".join(["?"] * len(play_ids))
        cur.execute(f"SELECT play_id, description, game_id, clock_display, player_id, player_name FROM plays WHERE play_id IN ({placeholders})", play_ids)
        play_rows = cur.fetchall()
        
        # Pull traits
        player_ids = list({r[4] for r in play_rows if r[4]})
        traits = {}
        if player_ids:
            ph2 = ",".join(["?"] * len(player_ids))
            cur.execute(f"SELECT player_id, dog_index, menace_index, unselfish_index, toughness_index, rim_pressure_index, shot_making_index, gravity_index, size_index FROM player_traits WHERE player_id IN ({ph2})", player_ids)
            for r in cur.fetchall():
                traits[r[0]] = {
                    "dog": r[1], "menace": r[2], "unselfish": r[3], "tough": r[4], 
                    "rim": r[5], "shot": r[6], "gravity": r[7], "size": r[8]
                }
        
        # Pull Matchups
        game_ids = list({r[2] for r in play_rows})
        matchups = {}
        if game_ids:
            ph3 = ",".join(["?"] * len(game_ids))
            cur.execute(f"SELECT game_id, home_team, away_team FROM games WHERE game_id IN ({ph3})", game_ids)
            matchups = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
            
        conn.close()

        # Build Rows
        rows = []
        for pid, desc, gid, clock, player_id, player_name in play_rows:
            t = traits.get(player_id, {})
            # Simple Filter Check
            if (t.get("dog") or 0) < slider_dog: continue
            
            home, away = matchups.get(gid, ("?", "?"))
            rows.append({
                "Player": player_name,
                "Player ID": player_id,
                "Match": f"{home} vs {away}",
                "Desc": desc,
                "Score": 0, # Simplify score for now
                "Traits": t
            })

        if rows:
            # Group by Player
            grouped = {}
            for r in rows:
                grouped.setdefault(r["Player"], []).append(r)
            
            # --- TOP 5 CARDS ---
            st.markdown("### Top Players")
            cols = st.columns(5)
            
            # Sort players by generic "fit" (placeholder logic)
            sorted_players = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
            
            for idx, (player_name, clips) in enumerate(sorted_players[:5]):
                clip = clips[0]
                pid = clip["Player ID"]
                t = clip["Traits"]
                
                with cols[idx]:
                    st.markdown(f"**{player_name}**")
                    st.caption(f"Dog Index: {t.get('dog',0):.0f}")
                    # THE FIX: Use Callback
                    st.button("View Profile", key=f"card_{pid}_{idx}", on_click=_go_to_player, args=(pid,))

            # --- LIST RESULTS ---
            st.markdown("### All Results")
            for player_name, clips in sorted_players:
                clip = clips[0]
                pid = clip["Player ID"]
                with st.expander(f"{player_name} ({len(clips)} clips)"):
                    # THE FIX: Use Callback
                    st.button("Go to Profile", key=f"list_{pid}", on_click=_go_to_player, args=(pid,))
                    for c in clips:
                        st.text(f"{c['Match']}: {c['Desc']}")
        else:
            st.info("No matches found matching criteria.")
