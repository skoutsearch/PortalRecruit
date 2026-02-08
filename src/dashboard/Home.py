import sys
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import zipfile
import json
import math
import re
import os
import base64
import time
from difflib import SequenceMatcher
import requests

# --- 1. SETUP PATHS ---
# Ensure repo root is on sys.path so imports work
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DB_PATH = REPO_ROOT / "data" / "skout.db"

from config.ncaa_di_mens_basketball import NCAA_DI_MENS_BASKETBALL
from src.ingestion.db import connect_db, ensure_schema

# Ensure DB schema (incl. social scout tables) exists on app startup
try:
    _conn = connect_db()
    ensure_schema(_conn)
    _conn.close()
except Exception:
    pass

# --- 2. PAGE CONFIGURATION ---
def get_base64_image(image_path):
    """Encodes a local image to base64 for embedding in HTML."""
    try:
        # Resolve full path relative to REPO_ROOT for safety
        full_path = REPO_ROOT / image_path
        with open(full_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

from src.dashboard.theme import inject_background

st.set_page_config(
    page_title="PortalRecruit | AI Scouting",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
css_path = REPO_ROOT / "www" / "streamlit.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    try:
        with open("streamlit.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

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
    name = (name or "").strip()
    if not name:
        return None

    cols = _players_table_columns()
    if not cols:
        return None

    import sqlite3
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        # Exact match first (fast + deterministic)
        cur.execute(f"SELECT {cols['id']} FROM players WHERE {cols['name']} = ? LIMIT 1", (name,))
        row = cur.fetchone()
        if row and row[0] is not None:
            con.close()
            return _normalize_player_id(row[0])

        # Fallback: case-insensitive match (helps when upstream title-cases, etc.)
        cur.execute(
            f"SELECT {cols['id']} FROM players WHERE LOWER({cols['name']}) = LOWER(?) LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        con.close()
        return _normalize_player_id(row[0]) if row and row[0] is not None else None
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        return None



@st.cache_resource(show_spinner=False)
def _get_search_collection():
    import chromadb

    vector_db_path = REPO_ROOT / "data" / "vector_db"
    client = chromadb.PersistentClient(path=str(vector_db_path))
    try:
        return client.get_collection(name="skout_plays")
    except Exception:
        _restore_vector_db_if_needed()
        client = chromadb.PersistentClient(path=str(vector_db_path))
        return client.get_collection(name="skout_plays")


@st.cache_data(show_spinner=False, max_entries=50000)
def _tag_play_cached(description: str) -> tuple[str, ...]:
    from src.processing.play_tagger import tag_play

    return tuple(tag_play(description or ""))


def _required_tag_threshold(required_tags: list[str]) -> int:
    n = len(set([t for t in required_tags if t]))
    if n <= 1:
        return n
    if n == 2:
        return 1
    return 2


def _get_qp():
    try:
        return st.query_params
    except Exception:
        return {}


def _set_qp(**kwargs):
    for k, v in kwargs.items():
        st.query_params[k] = v


def _clear_qp(key):
    try:
        if key in st.query_params:
            del st.query_params[key]
    except Exception:
        pass

def _get_player_profile(player_id: str):
    """Fetch player profile with traits, stats, and plays."""
    import sqlite3

    pid = _normalize_player_id(player_id)
    if not pid:
        return None

    cols = _players_table_columns()
    if not cols:
        return None

    def fetch_by_id(cur, pid_value):
        cur.execute(
            "SELECT {id_col}, {name_col}, {pos_col}, {team_col}, {class_col}, {ht_col}, {wt_col}, {hs_col} FROM players WHERE {id_col} = ? LIMIT 1".format(
                id_col=cols["id"],
                name_col=cols["name"],
                pos_col=cols["position"] or "NULL",
                team_col=cols["team_id"] or "NULL",
                class_col=cols["class_year"] or "NULL",
                ht_col=cols["height_in"] or "NULL",
                wt_col=cols["weight_lb"] or "NULL",
                hs_col=cols["high_school"] or "NULL",
            ),
            (pid_value,),
        )
        return cur.fetchone()

    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        row = fetch_by_id(cur, pid)
        if not row and pid.isdigit():
            row = fetch_by_id(cur, int(pid))
        if not row and player_id is not None:
            row = fetch_by_id(cur, str(player_id))
        if not row:
            con.close()
            return None

        profile = {
            "player_id": _normalize_player_id(row[0]),
            "name": row[1],
            "position": row[2] if cols.get("position") else None,
            "team_id": row[3] if cols.get("team_id") else None,
            "class_year": row[4] if cols.get("class_year") else None,
            "height_in": row[5] if cols.get("height_in") else None,
            "weight_lb": row[6] if cols.get("weight_lb") else None,
            "high_school": row[7] if cols.get("high_school") else None,
        }

        # traits
        cur.execute("SELECT * FROM player_traits WHERE player_id = ?", (pid,))
        trow = cur.fetchone()
        traits = {}
        if trow:
            cols_t = [d[0] for d in cur.description]
            traits = dict(zip(cols_t, trow))
        profile["traits"] = traits

        # season stats
        cur.execute(
            """
            SELECT season_id, season_label, gp, possessions, points, fg_made, shot3_made, ft_made,
                   fg_percent, shot3_percent, ft_percent, turnover,
                   minutes, reb, ast, stl, blk, ppg, rpg, apg
            FROM player_season_stats
            WHERE player_id = ?
            ORDER BY season_id DESC
            LIMIT 1
            """,
            (pid,),
        )
        srow = cur.fetchone()
        if srow:
            profile["stats"] = {
                "season_id": srow[0],
                "season_label": srow[1],
                "gp": srow[2],
                "possessions": srow[3],
                "points": srow[4],
                "fg_made": srow[5],
                "shot3_made": srow[6],
                "ft_made": srow[7],
                "fg_percent": srow[8],
                "shot3_percent": srow[9],
                "ft_percent": srow[10],
                "turnover": srow[11],
                "minutes": srow[12],
                "reb": srow[13],
                "ast": srow[14],
                "stl": srow[15],
                "blk": srow[16],
                "ppg": srow[17],
                "rpg": srow[18],
                "apg": srow[19],
            }
        else:
            profile["stats"] = {}

        # plays
        cur.execute("SELECT play_id, description, game_id, clock_display FROM plays WHERE player_id = ? LIMIT 25", (pid,))
        plays = cur.fetchall()
        profile["plays"] = plays

        # matchups + video
        game_ids = list({p[2] for p in plays})
        matchups = {}
        if game_ids:
            ph = ",".join(["?"] * len(game_ids))
            cur.execute(f"SELECT game_id, home_team, away_team, video_path FROM games WHERE game_id IN ({ph})", game_ids)
            matchups = {r[0]: (r[1], r[2], r[3]) for r in cur.fetchall()}
        profile["matchups"] = matchups

        con.close()
        return profile
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        return None


def _scout_breakdown(profile: dict) -> str:
    name = profile.get("name", "Player")
    traits = profile.get("traits", {}) or {}
    stats = profile.get("stats", {}) or {}
    position = profile.get("position") or ""
    height = _fmt_height(profile.get("height_in")) if profile.get("height_in") else ""
    school = profile.get("team_id") or ""

    strengths = []
    weaknesses = []
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

    ppg = stats.get("ppg")
    rpg = stats.get("rpg")
    apg = stats.get("apg")

    lines = []
    intro = f"{name}"
    if position:
        intro += f" ({position}" + (f", {height}" if height else "") + ")"
    if school:
        intro += f" at {school}"
    lines.append(intro + ".")

    if ppg is not None and rpg is not None and apg is not None:
        lines.append(f"Production snapshot: {ppg:.1f} PPG, {rpg:.1f} RPG, {apg:.1f} APG.")

    if strengths:
        lines.append(f"Strengths: {', '.join(strengths[:3])} show up consistently on film and in the data.")
    if weaknesses:
        lines.append(f"Growth areas: {', '.join(weaknesses[:2])}.")
    if not strengths and not weaknesses:
        lines.append("Profiles as a balanced contributor with a mix of skill and competitive traits.")

    lines.append("Overall, the profile suggests a dependable contributor who can fit into winning lineups when used correctly.")
    return " ".join(lines)


def _llm_scout_breakdown(profile):
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        # richer fallback
        name = profile.get("name", "Player")
        traits = profile.get("traits", {}) or {}
        strengths = []
        weaknesses = []
        for key, label in [
            ("dog_index", "dog mentality"),
            ("menace_index", "defensive menace"),
            ("unselfish_index", "unselfish playmaking"),
            ("toughness_index", "tough, competitive edge"),
            ("rim_pressure_index", "rim pressure"),
            ("shot_making_index", "shot making"),
            ("size_index", "size/length"),
        ]:
            val = traits.get(key)
            if val is None:
                continue
            if val >= 70:
                strengths.append(label)
            elif val <= 35:
                weaknesses.append(label)
        s = ", ".join(strengths[:3]) if strengths else "balanced skill mix"
        w = ", ".join(weaknesses[:2]) if weaknesses else "no glaring red flags"
        return (
            f"{name} brings a coachable, competitive profile with {s}. "
            f"Plays with good feel and shows the kind of habits that translate to winning possessions. "
            f"Growth areas to monitor: {w}."
        )

    name = profile.get("name", "Player")
    traits = profile.get("traits", {}) or {}
    stats = profile.get("stats", {}) or {}
    season_label = stats.get("season_label") or stats.get("season_id") or ""
    search_tags = profile.get("search_tags", []) or []
    plays = profile.get("plays", [])[:6]
    clips = []
    for play_id, desc, game_id, clock in plays:
        if desc:
            clips.append({"id": play_id, "desc": desc})

    model_name = os.getenv("OPENAI_MODEL") or st.secrets.get("OPENAI_MODEL") or "gpt-5-nano"

    # force specificity with unique facts
    fact_bits = []
    if stats.get("ppg") is not None:
        fact_bits.append(f"{stats.get('ppg'):.1f} PPG")
    if stats.get("rpg") is not None:
        fact_bits.append(f"{stats.get('rpg'):.1f} RPG")
    if stats.get("apg") is not None:
        fact_bits.append(f"{stats.get('apg'):.1f} APG")
    if profile.get("height_in"):
        fact_bits.append(f"Height {profile.get('height_in')} in")
    if profile.get("weight_lb"):
        fact_bits.append(f"Weight {profile.get('weight_lb')} lb")
    if profile.get("class_year"):
        fact_bits.append(f"Class {profile.get('class_year')}")
    if profile.get("team_id"):
        fact_bits.append(f"School {profile.get('team_id')}")

    prompt = f"""
You are The Old Recruiter — a 25+ year college basketball scout. Produce a premium, structured player breakdown for {name}.
Be specific, grounded in the data, and use coach-speak. Keep it clean and readable.

STRICT REQUIREMENTS:
- Mention at least 3 UNIQUE facts from this list: {fact_bits}
- Reference the season label when citing production.
- If you cannot comply, reply ONLY: INSUFFICIENT

FORMAT (use headings and bullets):
1) Snapshot (1–2 lines)
2) Strengths (bullet list)
3) Weaknesses / Growth Areas (bullet list)
4) Role & Projection (short paragraph)
5) Search Tag Fit (bullet list of the most relevant tags)
6) Old Recruiter Summary (1–2 strong, human paragraphs — colloquial, realistic, and decisive)
7) Clip Notes (reference 3–5 clips with citations like [clip:ID] if available)

Season: {season_label}
Traits: {traits}
Stats: {stats}
Search Tags: {search_tags}
Clips: {clips}
""".strip()

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a veteran college basketball recruiter."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 520,
            },
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        if content.strip() == "INSUFFICIENT":
            return _scout_breakdown(profile)
        return content
    except Exception:
        return _scout_breakdown(profile)


def _build_video_search_query(profile: dict) -> str:
    name = profile.get("name") or ""
    school = profile.get("team_id") or ""
    hs = profile.get("high_school") or ""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL") or st.secrets.get("OPENAI_MODEL") or "gpt-4o-mini"

    heuristic = f"\"{name}\" (\"{school}\" OR {school}) (\"{hs}\" OR {hs}) (basketball OR highlights OR mixtape) site:youtube.com"

    if not api_key:
        return heuristic

    prompt = f"""
Create one highly targeted Google search query to find YouTube basketball clips for the player.
Use expert boolean operators, quotes, and parentheses. Use the player's name plus school or high school.
Return ONLY the query string.
Player: {name}
School: {school}
High School: {hs}
""".strip()

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert search operator."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 80,
            },
            timeout=20,
        )
        resp.raise_for_status()
        query = resp.json()["choices"][0]["message"]["content"].strip()
        return query or heuristic
    except Exception:
        return heuristic


def _serper_video_search(query: str, num: int = 6) -> list[dict]:
    api_key = os.getenv("SERPER_API_KEY") or st.secrets.get("SERPER_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://google.serper.dev/videos",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        return data.get("videos", []) or []
    except Exception:
        return []


def _get_fallback_videos(profile: dict) -> list[str]:
    pid = _normalize_player_id(profile.get("player_id") or profile.get("id"))
    cache_key = f"fallback_videos_{pid}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached

    query = _build_video_search_query(profile)
    results = _serper_video_search(query)
    urls = []
    for r in results:
        link = r.get("link") or r.get("url")
        if link and "youtube.com" in link:
            urls.append(link)
    st.session_state[cache_key] = urls
    return urls



def _render_profile_overlay(player_id: str):
    pid = _normalize_player_id(player_id)
    profile = _get_player_profile(pid)

    if profile is not None:
        profile["search_tags"] = st.session_state.get("last_query_tags", []) or []

    # Fallback: use metadata captured from the current search results
    if not profile:
        cache = st.session_state.get("player_meta_cache", {}) or {}
        meta = cache.get(pid) if pid else None
        if meta:
            profile = {
                "player_id": pid,
                "name": meta.get("name") or meta.get("player") or "Player Profile",
                "position": meta.get("position"),
                "team_id": meta.get("team_id") or meta.get("team"),
                "class_year": meta.get("class_year"),
                "height_in": meta.get("height_in") or meta.get("height"),
                "weight_lb": meta.get("weight_lb") or meta.get("weight"),
                "high_school": meta.get("high_school") or meta.get("hs"),
                "traits": meta.get("traits") or {},
                "search_tags": st.session_state.get("last_query_tags", []) or [],
            }

    if not profile:
        st.warning("Player not found.")
        return
    title = profile.get("name", "Player Profile")

    def body():
        st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                <h2 style="margin:0; padding:0; color:white; font-size:2rem;">{title}</h2>
            </div>
        """, unsafe_allow_html=True)

        cache = st.session_state.get("player_meta_cache", {}) or {}
        meta_cache = cache.get(pid, {}) if pid else {}
        score = meta_cache.get("score")

        # Resolve team label
        team_label = str(profile.get("team_id") or "Unknown")
        if len(team_label) > 16 and team_label.replace("-", "").isalnum() and " " not in team_label:
            try:
                if "team_name_by_id" in locals():
                    team_label = team_name_by_id.get(team_label, team_label)
            except Exception:
                pass
        if len(team_label) > 16 and team_label.replace("-", "").isalnum() and " " not in team_label:
            team_label = "Unknown"

        # Header card
        stats = profile.get("stats", {}) or {}
        season_label = stats.get("season_label") or stats.get("season_id") or ""
        ppg = stats.get("ppg")
        rpg = stats.get("rpg")
        apg = stats.get("apg")

        height = _fmt_height(profile.get("height_in")) if profile.get("height_in") else "—"
        weight = f"{int(profile.get('weight_lb'))} lbs" if profile.get("weight_lb") else "—"
        hs = profile.get("high_school") or "—"
        class_year = profile.get("class_year") or "—"
        position = profile.get("position") or "—"
        score_tag = f"Recruit Score {score:.1f}" if score is not None else "Recruit Score —"

        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
                        border:1px solid rgba(255,255,255,0.08); padding:18px 20px; border-radius:16px;">
              <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
                <div>
                  <div style="font-size:1.05rem; letter-spacing:0.5px; text-transform:uppercase; opacity:0.7;">{team_label}</div>
                  <div style="font-size:2.1rem; font-weight:700; color:white; margin-top:2px;">{title}</div>
                  <div style="margin-top:6px; opacity:0.85;">{position} • {class_year} • {height} • {weight}</div>
                  <div style="margin-top:6px; opacity:0.7; font-size:0.95rem;">HS: {hs}</div>
                </div>
                <div style="text-align:right; min-width:180px;">
                  <div style="font-size:0.9rem; opacity:0.7;">{season_label} Production</div>
                  <div style="font-size:1.4rem; font-weight:600;">{(ppg if ppg is not None else 0):.1f} PPG</div>
                  <div style="opacity:0.8;">{(rpg if rpg is not None else 0):.1f} RPG • {(apg if apg is not None else 0):.1f} APG</div>
                  <div style="margin-top:8px; padding:6px 10px; display:inline-block; border-radius:999px; border:1px solid rgba(255,255,255,0.15); font-size:0.85rem;">{score_tag}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        stats = profile.get("stats", {}) or {}
        if stats:
            st.markdown("### Stats Snapshot")
            cols = st.columns(4)
            def _pct(v):
                return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "—"
            def _val(v):
                return "—" if v is None else v
            cols[0].metric("GP", _val(stats.get("gp")))
            cols[1].metric("PTS", _val(stats.get("points")))
            cols[2].metric("REB", _val(stats.get("reb")))
            cols[3].metric("AST", _val(stats.get("ast")))
            cols = st.columns(4)
            cols[0].metric("STL", _val(stats.get("stl")))
            cols[1].metric("BLK", _val(stats.get("blk")))
            cols[2].metric("MIN", _val(stats.get("minutes")))
            cols[3].metric("TOV", _val(stats.get("turnover")))
            cols = st.columns(3)
            cols[0].metric("FG%", _pct(stats.get("fg_percent")))
            cols[1].metric("3P%", _pct(stats.get("shot3_percent")))
            cols[2].metric("FT%", _pct(stats.get("ft_percent")))
            cols = st.columns(3)
            cols[0].metric("PPG", _val(stats.get("ppg")))
            cols[1].metric("RPG", _val(stats.get("rpg")))
            cols[2].metric("APG", _val(stats.get("apg")))

        st.markdown("### Scout Breakdown")
        breakdown = _llm_scout_breakdown(profile)
        breakdown = re.sub(r"\[clip:(\d+)\]", r"[clip](#clip-\1)", breakdown)
        st.markdown(
            f"<div style='background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25); padding:12px 14px; border-radius:10px; color:#e5e7eb;'>" + breakdown + "</div>",
            unsafe_allow_html=True,
        )

        # tags applied
        from src.processing.play_tagger import tag_play
        plays = profile.get("plays", [])
        tag_counts = {}
        for _, desc, _, _ in plays:
            for t in _tag_play_cached(desc):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        if tag_counts:
            st.markdown("### Tags Applied")
            tags_sorted = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
            st.write(", ".join([t for t, _ in tags_sorted[:12]]))

        # traits + strengths/weaknesses
        traits = profile.get("traits", {}) or {}
        if traits:
            st.markdown("### Strengths / Weaknesses")
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
            if strengths or weaknesses:
                st.markdown("**ML Assessment**")
            if strengths:
                st.success(f"**Strengths:** {', '.join(strengths[:4])}")
            if weaknesses:
                st.error(f"**Weaknesses:** {', '.join(weaknesses[:3])}")
            tags = profile.get("search_tags", []) or []
            if tags:
                st.markdown("**Matched Tags**")
                st.write("• " + "\n• ".join(tags))

        # clips section
        matchups = profile.get("matchups", {})
        last_tags = set(st.session_state.get("last_query_tags", []) or [])
        filtered = []
        if plays:
            for play_id, desc, game_id, clock in plays:
                tags = set(_tag_play_cached(desc))
                if last_tags and not tags.intersection(last_tags):
                    continue
                filtered.append((play_id, desc, game_id, clock, tags))

        clips = filtered if filtered else [(p[0], p[1], p[2], p[3], set(_tag_play_cached(p[1]))) for p in plays]
        st.markdown("### Film Room")
        if clips:
            # Build unique video list
            vid_items = []
            for play_id, desc, game_id, clock, tags in clips:
                home, away, video = matchups.get(game_id, ("Unknown", "Unknown", None))
                if video:
                    vid_items.append({
                        "play_id": play_id,
                        "desc": desc,
                        "game": f"{home} vs {away}",
                        "clock": clock,
                        "video": video,
                        "tags": tags,
                    })
            # de-dupe by video url
            uniq = []
            seen = set()
            for v in vid_items:
                if v["video"] in seen:
                    continue
                seen.add(v["video"])
                uniq.append(v)

            top3 = uniq[:3]
            if top3:
                cols = st.columns(3)
                for i, v in enumerate(top3):
                    with cols[i % 3]:
                        st.markdown(f"**{v['game']}**")
                        st.caption(v["clock"])
                        st.video(v["video"], start_time=0)
                if len(uniq) > 3:
                    if st.button("See more", key=f"see_more_{pid}"):
                        st.session_state[f"show_more_videos_{pid}"] = True
            else:
                fallback = _get_fallback_videos(profile)
                if fallback:
                    st.caption("No Synergy clips available — showing web‑found video highlights.")
                    cols = st.columns(3)
                    for i, vurl in enumerate(fallback[:3]):
                        with cols[i % 3]:
                            st.video(vurl)
                    if len(fallback) > 3:
                        st.button("See more", key=f"see_more_web_{pid}")
                else:
                    st.caption("No video clips available.")

            if st.session_state.get(f"show_more_videos_{pid}"):
                st.markdown("#### All Relevant Clips")
                cols = st.columns(3)
                for i, v in enumerate(uniq):
                    with cols[i % 3]:
                        st.markdown(f"**{v['game']}**")
                        st.caption(v["clock"])
                        st.video(v["video"], start_time=0)
                if st.button("Close", key=f"close_more_{pid}"):
                    st.session_state[f"show_more_videos_{pid}"] = False
        else:
            fallback = _get_fallback_videos(profile)
            if fallback:
                st.caption("No Synergy clips available — showing web‑found video highlights.")
                cols = st.columns(3)
                for i, vurl in enumerate(fallback[:3]):
                    with cols[i % 3]:
                        st.video(vurl)
                if len(fallback) > 3:
                    st.button("See more", key=f"see_more_web_{pid}")
            else:
                st.caption("No clips available for this player.")

        # Social media scouting report (Auto-Scout)
        st.markdown("### Social Media Scouting Report")
        report = _get_social_report(pid)
        queue_status = _get_social_queue_status(pid)

        # Progress indicator
        progress_val = 0
        if queue_status:
            if queue_status.get("status") == "queued":
                progress_val = 25
            elif queue_status.get("status") == "running":
                progress_val = 60
            elif queue_status.get("status") == "done":
                progress_val = 100
            elif queue_status.get("status") == "error":
                progress_val = 0
        if progress_val:
            st.progress(progress_val)

        if report and report.get("status") == "complete":
            rep = report.get("report") or {}
            handle = rep.get("verified_handle") or report.get("handle") or ""
            platform = rep.get("platform") or report.get("platform") or ""
            confidence = rep.get("confidence", "—")
            vibe = rep.get("vibe_check", "—")
            green = rep.get("green_flags") or []
            red = rep.get("red_flags") or []
            persona = rep.get("persona_tags") or []
            leadership = rep.get("leadership_signals") or []
            discipline = rep.get("discipline_concerns") or []
            nil_ops = rep.get("NIL_opportunities") or []
            risk = rep.get("recruiting_risk") or "—"
            summary = rep.get("summary") or ""
            recommendation = rep.get("recommendation") or ""

            st.markdown(
                f"**Verified:** {handle} {f'({platform})' if platform else ''} — **Confidence:** {confidence}%**" 
                if handle else "**Verified:** —"
            )
            if vibe:
                st.success(f"Vibe Check: {vibe}")
            st.markdown(f"**Recruiting Risk:** {risk}")
            if persona:
                st.markdown("**Persona Tags**")
                st.write("• " + "\n• ".join(persona))
            if green:
                st.markdown("**Green Flags**")
                st.write("• " + "\n• ".join(green))
            if red:
                st.markdown("**Red Flags**")
                st.write("• " + "\n• ".join(red))
            if leadership:
                st.markdown("**Leadership Signals**")
                st.write("• " + "\n• ".join(leadership))
            if discipline:
                st.markdown("**Discipline Concerns**")
                st.write("• " + "\n• ".join(discipline))
            if nil_ops:
                st.markdown("**NIL Opportunities**")
                st.write("• " + "\n• ".join(nil_ops))
            if summary:
                st.info(summary)
            if recommendation:
                st.markdown("**Recruiting Recommendation**")
                st.write(recommendation)

            if report.get("chosen_url"):
                st.markdown(f"[Source Profile]({report['chosen_url']})")

        elif queue_status and queue_status.get("status") in {"queued", "running"}:
            st.caption("Scouting in progress… this may take a few minutes.")
        elif queue_status and queue_status.get("status") == "error":
            st.error(f"Scouting failed: {queue_status.get('last_error')}")

        if st.button("Generate Social Media Scouting Report", key=f"gen_social_{pid}"):
            _enqueue_social_report(pid)
            st.success("Scouting queued. Refresh in a couple minutes.")

        # video evidence (placeholder)
        st.markdown("### Video Evidence")
        video_links = [v for _, _, v in matchups.values() if v]
        if video_links:
            uniq = []
            for v in video_links:
                if v not in uniq:
                    uniq.append(v)
            for v in uniq[:8]:
                st.markdown(f"• [Video Evidence]({v})")
        else:
            st.caption("No video evidence linked yet — will populate when full-access API URLs are available.")

    dialog_fn = getattr(st, "dialog", None)

    if callable(dialog_fn):
        @dialog_fn("Player Profile")
        def show_dialog():
            cols = st.columns([1, 7])
            if cols[0].button("<", key="back_profile"):
                _clear_qp("player")
                st.rerun()
            body()
        show_dialog()
    else:
        st.markdown("---")
        cols = st.columns([1, 7])
        if cols[0].button("<", key="back_profile"):
            _clear_qp("player")
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
    banner_html = """
    <div style="display:flex; justify-content:center; margin-bottom:12px;">
         <img src="https://portalrecruit.github.io/PortalRecruit/PORTALRECRUIT_WORDMARK_V4.webp" style="max-width:92vw; width:680px; height:auto; object-fit:contain; display:block;">
    </div>
    """

    hero_html = f"""
    <div class="pr-hero">
      {banner_html}
      <div style="display:flex; justify-content:center; margin:-6px auto 0;">
        <img src="https://portalrecruit.github.io/PortalRecruit/PORTALRECRUIT_TAGLINE_SEARCH_RECRUIT_WIN.webp" style="max-width:92vw; width:680px; height:auto; object-fit:contain; opacity:0.95;" />
      </div>
    </div>
    """

    components.html(hero_html, height=360)


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def _fmt_height(height_in):
    try:
        inches = int(round(float(height_in)))
    except Exception:
        return "—"
    if inches <= 0:
        return "—"
    ft = inches // 12
    inch = inches % 12
    return f"{ft}'{inch}\""


def _build_social_search_query(profile: dict) -> str:
    name = (profile.get("name") or "").strip()
    school = (profile.get("team_id") or "").strip()
    hs = (profile.get("high_school") or "").strip()
    class_year = (profile.get("class_year") or "").strip()
    # Use class year only if present
    class_term = f"\"Class of {class_year}\"" if class_year else ""

    parts = []
    if name:
        parts.append(f"\"{name}\"")

    sites = "(site:instagram.com OR site:tiktok.com OR site:twitter.com OR site:x.com OR site:facebook.com)"
    parts.append(sites)

    validators = []
    if school:
        validators.append(f"\"{school}\"")
    if hs:
        validators.append(f"\"{hs}\"")
    if class_term:
        validators.append(class_term)
    validators.extend(["basketball", "athlete"])

    if validators:
        parts.append("(" + " OR ".join(validators) + ")")

    return " ".join([p for p in parts if p])


def _build_old_recruiter_subject(query: str, matched_phrases: list[str] | None) -> str:
    phrases = [p for p in (matched_phrases or []) if p]
    # prefer unique, short phrases
    uniq = []
    for p in phrases:
        if p not in uniq:
            uniq.append(p)
    phrases = uniq[:3]

    # normalize big-men wording
    normalized = []
    for p in phrases:
        p_low = p.lower()
        if "big men" in p_low or "big man" in p_low:
            normalized.append("athletic center")
        else:
            normalized.append(p)
    phrases = normalized

    if phrases:
        return "Heard you were looking for " + ", ".join(phrases) + "..."
    q = (query or "").strip()
    return f"Here's the report you requested regarding {q}..."


def _enqueue_social_report(player_id: str) -> None:
    import sqlite3
    from datetime import datetime

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO social_scout_queue (player_id, status, requested_at) VALUES (?, 'queued', ?)",
        (player_id, datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


def _get_social_report(player_id: str):
    import sqlite3

    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            "SELECT status, report_json, chosen_url, platform, handle, updated_at FROM social_scout_reports WHERE player_id = ?",
            (player_id,),
        )
        row = cur.fetchone()
        con.close()
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        return None

    if not row:
        return None
    status, report_json, chosen_url, platform, handle, updated_at = row
    try:
        report = json.loads(report_json) if report_json else {}
    except Exception:
        report = {}
    return {
        "status": status,
        "report": report,
        "chosen_url": chosen_url,
        "platform": platform,
        "handle": handle,
        "updated_at": updated_at,
    }


def _get_social_queue_status(player_id: str):
    import sqlite3

    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            "SELECT status, requested_at, started_at, finished_at, last_error FROM social_scout_queue WHERE player_id = ? ORDER BY requested_at DESC LIMIT 1",
            (player_id,),
        )
        row = cur.fetchone()
        con.close()
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        return None

    if not row:
        return None
    return {
        "status": row[0],
        "requested_at": row[1],
        "started_at": row[2],
        "finished_at": row[3],
        "last_error": row[4],
    }


def _norm_person_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[\.'\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    parts = [p for p in name.split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(parts)


def _normalize_player_id(pid):
    'Normalize player IDs from different sources for stable SQLite lookup.'
    if pid is None:
        return None
    try:
        # Handle streamlit query_params list values
        if isinstance(pid, (list, tuple)) and pid:
            pid = pid[0]
    except Exception:
        pass
    try:
        # Common case: numeric id accidentally parsed as float (e.g., 12345.0)
        if isinstance(pid, float):
            if pid.is_integer():
                return str(int(pid))
            return str(pid).strip()
    except Exception:
        pass
    s = str(pid).strip()
    if not s:
        return None
    # Strip ".0" if present from CSV/JSON float conversions
    if re.match(r"^\d+\.0$", s):
        s = s.split(".", 1)[0]
    return s


@st.cache_data(show_spinner=False)
def _players_table_columns():
    'Detect players table schema; returns resolved column names or None.'
    import sqlite3
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='players'")
        if not cur.fetchone():
            con.close()
            return None
        cur.execute("PRAGMA table_info(players)")
        cols = [r[1] for r in cur.fetchall()]
        con.close()
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        return None

    def pick(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    resolved = {
        "id": pick(["player_id", "id", "playerId", "playerID"]),
        "name": pick(["full_name", "name", "player_name", "playerName"]),
        "position": pick(["position", "pos"]),
        "team_id": pick(["team_id", "teamId", "team"]),
        "class_year": pick(["class_year", "class", "year"]),
        "height_in": pick(["height_in", "height", "height_inches"]),
        "weight_lb": pick(["weight_lb", "weight", "weight_lbs"]),
        "high_school": pick(["high_school", "hs", "highschool"]),
    }
    # Must have at least id + name to be useful
    if not resolved["id"] or not resolved["name"]:
        return None
    return resolved


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
    # Sidebar Logo
    sb_logo = get_base64_image("www/PR_LOGO_BBALL_SQUARE.jpg")
    if sb_logo:
        st.markdown(f"""
            <div style="text-align:center; margin-bottom:20px;">
                <img src="data:image/jpeg;base64,{sb_logo}" style="width:120px; border-radius:50%; border:2px solid rgba(255,255,255,0.1);">
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align:center; font-family:var(--font-heading);'>PORTAL RECRUIT</h3>", unsafe_allow_html=True)
    st.divider()

    mode = st.radio(
        "NAVIGATION", 
        ["Search", "Admin"], 
        index=0 if st.session_state.app_mode == "Search" else 1,
        label_visibility="visible"
    )
    st.session_state.app_mode = mode
    st.divider()

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
    qp = _get_qp()
    if "player" in qp and qp["player"]:
        raw_pid = qp["player"][0] if isinstance(qp["player"], list) else qp["player"]
        pid = _normalize_player_id(raw_pid)

        # If profile exists in DB or is available from last search metadata, show it.
        if _get_player_profile(pid) or (st.session_state.get("player_meta_cache", {}) or {}).get(pid):
            _render_profile_overlay(pid)
            st.stop()

        _clear_qp("player")
        st.warning("Player not found.")
    
    # --- SEARCH INTERFACE ---
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # Custom Search Container
    st.markdown(
        "<div style='display:flex; justify-content:center; margin:-14px 0 14px;'>"
        "<img src='https://portalrecruit.github.io/PortalRecruit/PORTALRECRUIT_TAGLINE_SEARCH_RECRUIT_WIN.webp' "
        "style='max-width:92vw; width:520px; height:auto; object-fit:contain; opacity:0.95;'/>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Search box + autocomplete
    def _mark_search_requested():
        # User hit enter in the search bar
        st.session_state["search_requested"] = True
        st.session_state["search_status"] = "Searching"
        st.session_state["search_started_at"] = __import__("time").time()

    last_q = st.session_state.get("last_query") or ""
    search_status = st.session_state.get("search_status") or "Search"

    cols = st.columns([5, 1.2], gap="small")
    with cols[0]:
        query = st.text_input(
            "Player Search",
            last_q,
            placeholder="e.g. 'Athletic wing who can finish at the rim'",
            label_visibility="collapsed",
            on_change=_mark_search_requested,
            key="search_query_input",
        )
    with cols[1]:
        if st.button(search_status, key="search_btn", use_container_width=True):
            st.session_state["search_requested"] = True
            st.session_state["search_status"] = "Searching"
            st.session_state["search_started_at"] = __import__("time").time()

    # Recent searches
    try:
        import json
        mem_path = REPO_ROOT / "data" / "search_memory.json"
        if mem_path.exists():
            memory = json.loads(mem_path.read_text())
            recent = list(reversed(memory.get("queries", [])))[:8]
            if not query and recent:
                cols = st.columns([1,2,1])
                with cols[1]:
                    st.caption(f"Try: \"{recent[0]}\"")
    except Exception:
        pass

    try:
        from src.search.autocomplete import suggest_rich  # noqa: E402
        suggestions = suggest_rich(query, limit=25)
    except Exception:
        suggestions = []
    
    if query:
        pass 
    elif suggestions:
        # If user is typing but hasn't submitted? 
        # Streamlit doesn't support 'typing' events easily without custom component
        pass

    def _render_results(rows, query_text):
        if rows:
            # Group by player (top 3 clips each)
            grouped = {}
            for r in rows:
                grouped.setdefault(r["Player"], []).append(r)

            progress_placeholder = st.session_state.get("progress_placeholder")
            if progress_placeholder is not None:
                subject_line = _build_old_recruiter_subject(query_text, st.session_state.get("last_matched_phrases") or [])
                progress_placeholder.markdown(
                    f"""
                    <div class='old-recuiter-final'>
                      <div><strong>From:</strong> &lt;The Old Recruiter&gt; <a href='mailto:theoldrecruiter@portalrecruit.com'>theoldrecruiter@portalrecruit.com</a></div>
                      <div><strong>Subject:</strong> {subject_line}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<h3 style='margin-top:40px;'>Top Prospects</h3>", unsafe_allow_html=True)

            for player, clips in grouped.items():
                pid = _normalize_player_id(clips[0].get("Player ID")) if clips else None
                score = clips[0].get("Score", 0) if clips else 0
                st.session_state.setdefault("player_meta_cache", {})
                if pid:
                    st.session_state["player_meta_cache"][pid] = {
                        "name": player,
                        "position": clips[0].get("Position", "") if clips else "",
                        "team": clips[0].get("Team", "") if clips else "",
                        "team_id": clips[0].get("Team", "") if clips else "",
                        "height": clips[0].get("Height") if clips else None,
                        "weight": clips[0].get("Weight") if clips else None,
                        "class_year": clips[0].get("Class") if clips else None,
                        "high_school": clips[0].get("High School") if clips else None,
                        "ppg": clips[0].get("PPG") if clips else None,
                        "rpg": clips[0].get("RPG") if clips else None,
                        "apg": clips[0].get("APG") if clips else None,
                        "score": score,
                    }

                pos = clips[0].get("Position") if clips else None
                team = clips[0].get("Team") if clips else None
                ht = clips[0].get("Height") if clips else None
                wt = clips[0].get("Weight") if clips else None
                pos = pos if pos not in [None, "", "None"] else "—"
                team = team if team not in [None, "", "None"] else "—"
                size = "—"
                if ht and wt:
                    size = f"{_fmt_height(ht)} / {int(wt)} lbs"
                elif ht:
                    size = f"{_fmt_height(ht)}"
                elif wt:
                    size = f"{int(wt)} lbs"

                detail_parts = [
                    player,
                    pos if pos and pos != "—" else "—",
                    _fmt_height(ht) if ht else "—",
                    f"{int(wt)} lbs" if wt else "—",
                    team if team and team != "—" else "Unknown",
                    f"Recruit Score: {score:.1f}",
                ]

                label = " | ".join(detail_parts)

                if pid and st.button(label, key=f"btn_{pid}", use_container_width=True):
                    st.query_params["player"] = pid
                    st.rerun()

                # Secondary line with HS + class + per-game stats (if available)
                class_line = clips[0].get("Class") if clips else None
                hs_line = clips[0].get("High School") if clips else None
                ppg_line = clips[0].get("PPG") if clips else None
                rpg_line = clips[0].get("RPG") if clips else None
                apg_line = clips[0].get("APG") if clips else None
                extra = []
                if class_line and class_line != "—":
                    extra.append(f"Class: {class_line}")
                if hs_line and hs_line != "—":
                    extra.append(f"HS: {hs_line}")
                if ppg_line is not None and rpg_line is not None and apg_line is not None:
                    extra.append(f"{ppg_line:.1f} / {rpg_line:.1f} / {apg_line:.1f} PPG/RPG/APG")
                if extra:
                    st.caption(" • ".join(extra))
        else:
            st.info("No results after filters.")

    # Name-aware search routing (skip while active search)
    if not st.session_state.get("search_requested"):
        name_resolution = _resolve_name_query(query)
        if name_resolution.get("mode") == "exact_single":
            _set_qp(player=name_resolution["matches"][0]["player_id"])
            st.rerun()
        elif name_resolution.get("mode") in {"exact_multi", "fuzzy_multi"}:
            st.markdown("### Did you mean")
            cols = st.columns(2)
            for i, p in enumerate(name_resolution["matches"]):
                with cols[i % 2]:
                    # Card style for 'Did you mean'
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:8px; margin-bottom:8px;">
                        <div style="font-weight:bold; color:white;">{p['full_name']}</div>
                        <div style="font-size:0.8em; opacity:0.7;">{p.get('team_id','')} • {p.get('position','')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("View Profile", key=f"didyoumean_{p['player_id']}"):
                        st.session_state.profile_player_id = p["player_id"]
                        _render_profile_overlay(st.session_state.profile_player_id)
                        st.stop()
            st.stop()

    # Persist last results when not actively searching
    if not st.session_state.get("search_requested") and st.session_state.get("last_rows"):
        _render_results(st.session_state.get("last_rows") or [], query or st.session_state.get("last_query") or "")

    def _render_results(rows, query_text):
        if rows:
            # Group by player (top 3 clips each)
            grouped = {}
            for r in rows:
                grouped.setdefault(r["Player"], []).append(r)

            progress_placeholder = st.session_state.get("progress_placeholder")
            if progress_placeholder is not None:
                subject_line = _build_old_recruiter_subject(query_text, st.session_state.get("last_matched_phrases") or [])
            progress_placeholder.markdown(
                    f"""
                    <div class='old-recuiter-final'>
                      <div><strong>From:</strong> &lt;The Old Recruiter&gt; <a href='mailto:theoldrecruiter@portalrecruit.com'>theoldrecruiter@portalrecruit.com</a></div>
                      <div><strong>Subject:</strong> {subject_line}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<h3 style='margin-top:40px;'>Top Prospects</h3>", unsafe_allow_html=True)

            for player, clips in grouped.items():
                pid = _normalize_player_id(clips[0].get("Player ID")) if clips else None
                score = clips[0].get("Score", 0) if clips else 0
                st.session_state.setdefault("player_meta_cache", {})
                if pid:
                    st.session_state["player_meta_cache"][pid] = {
                        "name": player,
                        "position": clips[0].get("Position", "") if clips else "",
                        "team": clips[0].get("Team", "") if clips else "",
                        "team_id": clips[0].get("Team", "") if clips else "",
                        "height": clips[0].get("Height") if clips else None,
                        "weight": clips[0].get("Weight") if clips else None,
                        "class_year": clips[0].get("Class") if clips else None,
                        "high_school": clips[0].get("High School") if clips else None,
                        "ppg": clips[0].get("PPG") if clips else None,
                        "rpg": clips[0].get("RPG") if clips else None,
                        "apg": clips[0].get("APG") if clips else None,
                        "score": score,
                    }

                pos = clips[0].get("Position") if clips else None
                team = clips[0].get("Team") if clips else None
                ht = clips[0].get("Height") if clips else None
                wt = clips[0].get("Weight") if clips else None
                pos = pos if pos not in [None, "", "None"] else "—"
                team = team if team not in [None, "", "None"] else "—"
                size = "—"
                if ht and wt:
                    size = f"{_fmt_height(ht)} / {int(wt)} lbs"
                elif ht:
                    size = f"{_fmt_height(ht)}"
                elif wt:
                    size = f"{int(wt)} lbs"

                detail_parts = [
                    player,
                    pos if pos and pos != "—" else "—",
                    _fmt_height(ht) if ht else "—",
                    f"{int(wt)} lbs" if wt else "—",
                    team if team and team != "—" else "Unknown",
                    f"Recruit Score: {score:.1f}",
                ]

                label = " | ".join(detail_parts)

                if pid and st.button(label, key=f"btn_{pid}", use_container_width=True):
                    st.query_params["player"] = pid
                    st.rerun()

                # Secondary line with HS + class + per-game stats (if available)
                class_line = clips[0].get("Class") if clips else None
                hs_line = clips[0].get("High School") if clips else None
                ppg_line = clips[0].get("PPG") if clips else None
                rpg_line = clips[0].get("RPG") if clips else None
                apg_line = clips[0].get("APG") if clips else None
                extra = []
                if class_line and class_line != "—":
                    extra.append(f"Class: {class_line}")
                if hs_line and hs_line != "—":
                    extra.append(f"HS: {hs_line}")
                if ppg_line is not None and rpg_line is not None and apg_line is not None:
                    extra.append(f"{ppg_line:.1f} / {rpg_line:.1f} / {apg_line:.1f} PPG/RPG/APG")
                if extra:
                    st.caption(" • ".join(extra))
        else:
            st.info("No results after filters.")

    if query and st.session_state.get("search_requested"):
        st.session_state["search_status"] = "Searching"
        if not st.session_state.get("search_started_at"):
            st.session_state["search_started_at"] = __import__("time").time()

        # Immediate progress feedback
        progress_placeholder = st.empty()
        st.session_state["progress_placeholder"] = progress_placeholder
        def _stage(msg, color="#7aa2f7"):
            progress_placeholder.markdown(
                f"<div class='old-recuiter-stage' style='color:{color}'>" + msg + "</div>",
                unsafe_allow_html=True,
            )
        _stage("Phoning the Old Recruiter...", "#7aa2f7")
        __import__("time").sleep(10)
        _stage("Dropping the Old Recruiter off at the airport...", "#9b7bff")
        __import__("time").sleep(5)
        explaining_start = __import__("time").time()
        _stage("Explaining Uber to the Old Recruiter...", "#f6c177")

        # Search vars (Advanced Filters removed but logic kept for defaults)
        slider_dog = slider_menace = slider_unselfish = slider_tough = 0
        slider_rim = slider_shot = slider_gravity = slider_size = 0
        intent_dog = intent_menace = intent_unselfish = intent_tough = 0
        intent_rim = intent_shot = intent_gravity = intent_size = 0
        n_results = 150
        intent_tags = []
        required_tags = []
        finishing_intent = False

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
        apply_exclude = any(tok in q_lower for tok in [" no ", "avoid", "without", "dont", "don't", "not "])
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

        st.session_state["last_matched_phrases"] = matched_phrases

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

        # User requested no filters
        required_tags = []
        st.session_state["last_query"] = query
        st.session_state["last_query_tags"] = []

        # --- VECTOR SEARCH ---
        import sqlite3

        DB_PATH = REPO_ROOT / "data" / "skout.db"

        try:
            collection = _get_search_collection()
        except Exception:
            st.error("Vector DB not found. Run embeddings first (generate_embeddings.py) to create 'skout_plays'.")
            st.stop()

        # Query expansion: add matched phrases to help retrieval
        from src.search.semantic import build_expanded_query, semantic_search

        expanded_query = build_expanded_query(query, matched_phrases)

        status = st.status("Searching…", expanded=False)
        status.update(state="running")
        st.markdown(
            "<script>document.body.classList.add('searching');</script>",
            unsafe_allow_html=True,
        )

        # Old Recruiter progress display already initiated above

        # Vector search
        play_ids = semantic_search(
            collection,
            query=query,
            n_results=n_results,
            extra_query_terms=matched_phrases,
            required_tags=required_tags,
        )
        # ensure at least 5s on explaining stage
        elapsed = __import__("time").time() - explaining_start
        if elapsed < 5:
            __import__("time").sleep(5 - elapsed)

        st.markdown(
            "<script>document.body.classList.remove('searching');</script>",
            unsafe_allow_html=True,
        )

        if not play_ids:
            status.update(state="complete", label="Search complete")
            st.markdown(
                "<script>document.body.classList.remove('searching');</script>",
                unsafe_allow_html=True,
            )
            st.warning("No results found.")
        else:
            status.update(state="complete", label="Search complete")
            st.markdown(
                "<script>document.body.classList.remove('searching');</script>",
                unsafe_allow_html=True,
            )
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
            player_meta_by_name = {}
            team_name_by_id = {}
            try:
                conn_meta = sqlite3.connect(DB_PATH)
                cur_meta = conn_meta.cursor()
                cur_meta.execute("SELECT player_id, full_name, position, team_id, height_in, weight_lb, class_year, high_school FROM players")
                for r in cur_meta.fetchall():
                    pid_norm = _normalize_player_id(r[0])
                    meta_row = {
                        "full_name": r[1] or "",
                        "position": r[2] or "",
                        "team_id": r[3] or "",
                        "height_in": r[4],
                        "weight_lb": r[5],
                        "class_year": r[6] or "",
                        "high_school": r[7] or "",
                    }
                    if pid_norm:
                        player_meta[pid_norm] = meta_row
                    name_key = _norm_person_name(r[1] or "")
                    if name_key:
                        player_meta_by_name[name_key] = meta_row

                # Build team_id -> name map from plays (offense/defense team labels)
                try:
                    cur_meta.execute(
                        """
                        SELECT team_id, offense_team, defense_team
                        FROM plays
                        WHERE team_id IS NOT NULL
                          AND (offense_team IS NOT NULL OR defense_team IS NOT NULL)
                        """
                    )
                    counts = {}
                    for tid, off_t, def_t in cur_meta.fetchall():
                        for t in (off_t, def_t):
                            if not t:
                                continue
                            key = (tid, t)
                            counts[key] = counts.get(key, 0) + 1
                    for (tid, name), cnt in sorted(counts.items(), key=lambda x: -x[1]):
                        if tid not in team_name_by_id:
                            team_name_by_id[tid] = name
                except Exception:
                    team_name_by_id = {}

                conn_meta.close()
            except Exception:
                player_meta = {}
                player_meta_by_name = {}
                team_name_by_id = {}

            # Preload season stats + player names + full traits for similarity
            player_stats = {}
            player_names = {}
            traits_all = {}
            player_team_guess = {}
            try:
                conn2 = sqlite3.connect(DB_PATH)
                cur2 = conn2.cursor()
                cur2.execute(
                    """
                    SELECT player_id, season_id, team_id, gp, possessions, points,
                           fg_percent, shot2_percent, shot3_percent, ft_percent,
                           fg_attempt, shot2_attempt, shot3_attempt, turnover,
                           ppg, rpg, apg
                    FROM player_season_stats
                    ORDER BY season_id DESC
                    """
                )
                for r in cur2.fetchall():
                    pid = r[0]
                    if pid in player_stats:
                        continue
                    player_stats[pid] = {
                        "season_id": r[1],
                        "team_id": r[2],
                        "gp": r[3],
                        "possessions": r[4],
                        "points": r[5],
                        "fg_percent": r[6],
                        "shot2_percent": r[7],
                        "shot3_percent": r[8],
                        "ft_percent": r[9],
                        "fg_attempt": r[10],
                        "shot2_attempt": r[11],
                        "shot3_attempt": r[12],
                        "turnover": r[13],
                        "ppg": r[14],
                        "rpg": r[15],
                        "apg": r[16],
                    }

                # Guess team name per player from play-by-play
                try:
                    cur2.execute(
                        """
                        SELECT p.player_id, p.offense_team, p.defense_team, p.is_home, p.game_id
                        FROM plays p
                        WHERE p.player_id IS NOT NULL
                        """
                    )
                    counts = {}
                    for pid, off_t, def_t, is_home, gid in cur2.fetchall():
                        if off_t:
                            counts[(pid, off_t)] = counts.get((pid, off_t), 0) + 1
                        if def_t:
                            counts[(pid, def_t)] = counts.get((pid, def_t), 0) + 1
                    # fallback using games if no offense/defense labels
                    if not counts:
                        cur2.execute(
                            """
                            SELECT p.player_id, p.is_home, g.home_team, g.away_team
                            FROM plays p
                            JOIN games g ON g.game_id = p.game_id
                            WHERE p.player_id IS NOT NULL AND g.home_team IS NOT NULL AND g.away_team IS NOT NULL
                            """
                        )
                        for pid, is_home, home, away in cur2.fetchall():
                            team = home if is_home else away
                            if team:
                                counts[(pid, team)] = counts.get((pid, team), 0) + 1
                    for (pid, team), cnt in sorted(counts.items(), key=lambda x: -x[1]):
                        if pid not in player_team_guess:
                            player_team_guess[pid] = team
                except Exception:
                    player_team_guess = {}
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

            # Stage 3: DB fetch + enrichment
            _stage("Confirming arrival of the Old Recruiter at Prospect's local gym...", "#7bdcb5")

            rows = []
            for pid, desc, gid, clock, player_id, player_name in play_rows:
                pid_norm = _normalize_player_id(player_id)
                meta = player_meta.get(pid_norm) if pid_norm else None
                if meta is None and player_name:
                    meta = player_meta_by_name.get(_norm_person_name(player_name))
                if meta is None:
                    continue  # skip players without roster/meta info
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

                play_tags = list(_tag_play_cached(desc))
                if "non_possession" in play_tags:
                    continue
                if apply_exclude and exclude_tags and set(play_tags).intersection(exclude_tags):
                    continue
                # Adaptive required-tag filter: avoid zero-result dead ends on sparse tag vocab.
                if required_tags:
                    req_threshold = _required_tag_threshold(required_tags)
                    req_hits = len(set(required_tags).intersection(set(play_tags)))
                    if req_hits < req_threshold:
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
                if pos:
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

                meta = {}
                pid_norm = _normalize_player_id(player_id)
                if "player_meta" in locals() and pid_norm:
                    meta = player_meta.get(pid_norm, {})
                if not meta and "player_meta_by_name" in locals():
                    meta = player_meta_by_name.get(_norm_person_name(player_name or ""), {})

                pos_val = meta.get("position", "")
                team_val = meta.get("team_id", "")
                ht_val = meta.get("height_in")
                wt_val = meta.get("weight_lb")
                class_val = meta.get("class_year", "")
                hs_val = meta.get("high_school", "")
                ppg_val = (player_stats.get(pid_norm) or {}).get("ppg") if pid_norm else None
                rpg_val = (player_stats.get(pid_norm) or {}).get("rpg") if pid_norm else None
                apg_val = (player_stats.get(pid_norm) or {}).get("apg") if pid_norm else None
                pos_val = pos_val if pos_val not in [None, "", "None"] else "—"
                team_val = team_val if team_val not in [None, "", "None"] else "—"
                class_val = class_val if class_val not in [None, "", "None"] else "—"
                hs_val = hs_val if hs_val not in [None, "", "None"] else "—"
                # Replace internal IDs with team name when possible
                if team_val != "—":
                    team_clean = str(team_val).strip()
                    if len(team_clean) > 16 and team_clean.replace("-", "").isalnum() and " " not in team_clean:
                        team_val = team_name_by_id.get(team_val, "—")
                if team_val == "—" and player_team_guess:
                    team_val = player_team_guess.get(player_id, "—")
                if team_val == "—":
                    team_val = "Unknown"

                # ACC-only filter removed per user request

                rows.append({
                    "Match": f"{home} vs {away}",
                    "Clock": clock,
                    "Player": (player_name or "Unknown"),
                    "Player ID": player_id,
                    "Position": pos_val,
                    "Team": team_val,
                    "Height": ht_val,
                    "Weight": wt_val,
                    "Class": class_val,
                    "High School": hs_val,
                    "PPG": ppg_val,
                    "RPG": rpg_val,
                    "APG": apg_val,
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

            # Stage 4: scoring/ranking
            _stage("Incoming email from <a href='mailto:theoldrecruiter@portalrecruit.com'>theoldrecruiter@portalrecruit.com</a>...", "#ff7eb6")

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

            st.session_state["search_status"] = "Search"
            st.session_state["search_requested"] = False
            st.session_state["last_rows"] = rows

            _render_results(rows, query)
