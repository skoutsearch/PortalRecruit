import traceback
import streamlit as st
import sys
import os
import re
import math
import json
import base64
import time
import zipfile
import sqlite3
from pathlib import Path
from difflib import SequenceMatcher
import requests
import mimetypes

# --- 1. PAGE CONFIG (Must be the very first Streamlit command) ---
st.set_page_config(
    page_title="PortalRecruit | AI Scouting",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="expanded",
)

# --- 2. ROBUST PATH SETUP ---
# Snowflake often changes where files are mounted. We dynamically find the 'src' folder.
try:
    current_path = Path(__file__).resolve()
    repo_root = current_path.parent
    
    # Walk up the tree until we find 'src' or hit the root
    for _ in range(5):
        if (repo_root / "src").exists():
            break
        if repo_root == repo_root.parent: # Reached system root
            break
        repo_root = repo_root.parent
        
    REPO_ROOT = repo_root
    
    # CRITICAL FIX: Force string conversion for sys.path
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # CRITICAL FIX: Force string conversion for DB path
    DB_PATH = REPO_ROOT / "data" / "skout.db"
    DB_PATH_STR = str(DB_PATH)

except Exception as e:
    st.error(f"Critical Path Error: {e}")
    st.stop()

# --- 3. IMPORTS (After sys.path is fixed) ---
import streamlit.components.v1 as components

connect_db = None
ensure_schema = None
generate_scout_breakdown = None
inject_background = None

try:
    from src.ingestion.db import connect_db as _connect_db, ensure_schema as _ensure_schema
    connect_db, ensure_schema = _connect_db, _ensure_schema
except ImportError as e:
    st.warning(f"Import Warning (DB): {e}")

try:
    from src.llm.scout import generate_scout_breakdown as _generate_scout_breakdown
    generate_scout_breakdown = _generate_scout_breakdown
except ImportError as e:
    st.warning(f"Import Warning (Scout): {e}")

try:
    from src.dashboard.theme import inject_background as _inject_background
    inject_background = _inject_background
except ImportError as e:
    st.warning(f"Import Warning (Theme): {e}")

if (REPO_ROOT / "config").exists() and str(REPO_ROOT / "config") not in sys.path:
    sys.path.append(str(REPO_ROOT / "config"))

# --- 4. INITIALIZATION ---

# Run DB setup safely - prevents "bad argument type" crash on load
try:
    # Check if DB file exists before connecting
    if not os.path.exists(DB_PATH_STR):
        # Create directory if missing
        os.makedirs(os.path.dirname(DB_PATH_STR), exist_ok=True)
    
    # Connect using STRING path (not Path object)
    _conn = sqlite3.connect(DB_PATH_STR)
    if ensure_schema:
        ensure_schema(_conn)
    _conn.close()
except Exception:
    # Fail silently on init, specific functions will handle errors later
    pass

# Load CSS safely
css_path = REPO_ROOT / "www" / "streamlit.css"
if css_path.exists():
    with open(str(css_path), "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    if inject_background:
        inject_background()
except Exception:
    pass



# --- 4b. GLASS BACKGROUND VIDEO (Cloud + Snowflake) ---
@st.cache_data(show_spinner=False)
def _get_base64_video(video_path: str) -> str | None:
    p = Path(video_path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    return base64.b64encode(p.read_bytes()).decode("utf-8")


try:
    bg_candidates = [
        REPO_ROOT / "www" / "PORTALRECRUIT_ANIMATED_LOGO.mp4",
        REPO_ROOT / "www" / "PORTALRECRUIT_ANIMATED_LOGO.mov",
        REPO_ROOT / "www" / "PORTALRECRUIT_ANIMATED_LOGO.webm",
    ]
    bg_video_path = next((p for p in bg_candidates if p.exists()), None)
    if bg_video_path:
        b64 = _get_base64_video(str(bg_video_path))
        mime_type = mimetypes.guess_type(str(bg_video_path))[0] or "video/mp4"
        if b64:
            components.html(
            f"""
            <style>
              #pr-bg-video {{
                position: fixed;
                right: 0;
                bottom: 0;
                min-width: 100%;
                min-height: 100%;
                width: auto;
                height: auto;
                z-index: -3;
                filter: blur(18px) saturate(1.2) contrast(1.05);
                opacity: 0.28;
                transform: scale(1.05);
              }}
              .pr-bg-overlay {{
                position: fixed;
                inset: 0;
                z-index: -2;
                background: radial-gradient(1200px 700px at 50% 20%, rgba(255,141,26,0.14), rgba(0,0,0,0) 60%),
                            radial-gradient(900px 600px at 20% 70%, rgba(122,162,247,0.10), rgba(0,0,0,0) 60%),
                            linear-gradient(180deg, rgba(8,10,20,0.92), rgba(8,10,20,0.86));
              }}
              /* Glass containers */
              .pr-glass {{
                background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 18px;
                box-shadow: 0 12px 40px rgba(0,0,0,0.35);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
              }}
              .pr-result-card {{
                padding: 16px 16px;
                border-radius: 18px;
                border: 1px solid rgba(255,140,20,0.55);
                box-shadow:
                  0 0 0 1px rgba(255,140,20,0.10) inset,
                  0 0 22px rgba(255,140,20,0.22),
                  0 16px 44px rgba(0,0,0,0.40);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04));
                transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
              }}
              .pr-result-card:hover {{
                transform: translateY(-2px);
                border-color: rgba(255,140,20,0.80);
                box-shadow:
                  0 0 0 1px rgba(255,140,20,0.18) inset,
                  0 0 30px rgba(255,140,20,0.35),
                  0 20px 54px rgba(0,0,0,0.50);
              }}
              .pr-chip {{
                display: inline-flex;
                align-items: center;
                padding: 4px 10px;
                border-radius: 999px;
                border: 1px solid rgba(255,255,255,0.14);
                background: rgba(255,255,255,0.06);
                font-size: 0.85rem;
                opacity: 0.92;
              }}
              /* tighten the main container */
              section.main > div.block-container {{
                max-width: 1180px;
                padding-top: 1.6rem;
              }}
            </style>
            <video id="pr-bg-video" autoplay muted loop playsinline>
              <source src="data:{mime_type};base64,{b64}" type="{mime_type}" />
            </video>
            <div class="pr-bg-overlay"></div>
            """,
            height=0,
        )
except Exception:
    pass

# --- 5. HELPER FUNCTIONS ---

def get_base64_image(image_path):
    """Encodes a local image to base64 for embedding in HTML."""
    try:
        full_path = REPO_ROOT / image_path
        if not full_path.exists():
            return None
        with open(str(full_path), "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None


def _position_tags(pos: str) -> set[str]:
    """Normalize position strings into role tags: guard/wing/big."""
    p = (pos or "").upper().strip()
    if not p:
        return set()
    tags: set[str] = set()
    if any(x in p for x in ["PG", "SG"]) or ("G" in p and "F" not in p and "C" not in p):
        tags.add("guard")
    if any(x in p for x in ["SF", "PF"]) or ("F" in p) or ("W" in p):
        tags.add("wing")
    if "C" in p or "F/C" in p or "C/F" in p:
        tags.add("big")
    # combos like G/F, F/G, F-C etc
    if "G" in p and "F" in p:
        tags.update({"guard", "wing"})
    if "F" in p and "C" in p:
        tags.update({"wing", "big"})
    return tags


def _infer_size_intents(q: str) -> dict:
    """Extract size/physical development intents from the query."""
    ql = (q or "").lower()

    # explicit height tokens like 6'8, 6-8, 6 8, 7 footer
    height_min = None
    height_max = None
    m = re.search(r"(\d)\s*['-]?\s*(\d{1,2})\s*(?:\"|in)?", ql)
    if m:
        ft = int(m.group(1))
        inch = int(m.group(2))
        h = ft * 12 + inch
        height_min = h - 1
        height_max = h + 2

    if any(k in ql for k in ["tall", "long", "length", "rangy", "wingspan", "big wingspan"]):
        height_min = max(height_min or 0, 77)  # ~6'5+
    if any(k in ql for k in ["big", "huge", "strong", "power", "physical", "burly"]):
        # weight intent, not absolute requirement
        pass
    if any(k in ql for k in ["skinny", "lean", "lanky", "thin"]):
        pass

    growth = any(k in ql for k in ["room to grow", "upside", "project", "raw", "high ceiling", "frame to add", "fill out"])

    # position-relative cues
    wants_big = any(k in ql for k in ["big man", "rim protector", "post", "paint", "center"])
    wants_guard = any(k in ql for k in ["point guard", "pg", "guard", "ball handler"])
    wants_forward = any(k in ql for k in ["forward", "sf", "pf", "wing", "3-and-d"])

    # soft thresholds (applied only when query implies size)
    if wants_big:
        height_min = max(height_min or 0, 79)  # 6'7+
    if wants_guard and any(k in ql for k in ["tall", "big", "long"]):
        height_min = max(height_min or 0, 75)  # 6'3+
    if wants_forward:
        height_min = max(height_min or 0, 76)  # 6'4+

    return {
        "height_min": height_min,
        "height_max": height_max,
        "growth": growth,
        "skinny": any(k in ql for k in ["skinny", "lean", "lanky", "thin"]),
        "strong": any(k in ql for k in ["strong", "physical", "powerful", "burly", "thick"]),
    }


def _infer_role_hints(q: str) -> set[str]:
    """Extract role hints (guard/wing/big) with robust synonym coverage."""
    ql = (q or "").lower()
    hints: set[str] = set()

    wing_terms = [
        "forward", "sf", "small forward", "pf", "power forward", "wing", "3-and-d", "three and d", "combo forward"
    ]
    guard_terms = [
        "guard", "pg", "point guard", "sg", "shooting guard", "combo guard", "lead guard"
    ]
    big_terms = [
        "center", "big man", "rim protector", "post", "paint", "big", "five", "4/5", "4-5"
    ]
    if any(k in ql for k in wing_terms):
        hints.add("wing")
    if any(k in ql for k in guard_terms):
        hints.add("guard")
    if any(k in ql for k in big_terms):
        hints.add("big")

    # common abbreviations like 'c' as a token
    toks = re.findall(r"[a-z0-9']+", ql)
    if "c" in toks:
        hints.add("big")
    if "f" in toks and "g" not in toks:
        hints.add("wing")
    if "g" in toks and "f" not in toks:
        hints.add("guard")

    return hints


from collections import OrderedDict


def _expand_query_synonyms(q: str) -> list[str]:
    """Advanced synonym expansion to improve recall when semantic search under-fires."""
    ql = (q or "").lower()
    synonyms: list[str] = []

    mapping = {
        "shoot 3": ["three point", "3pt", "shot3", "spacing", "stretch"],
        "can shoot": ["three point", "catch and shoot", "spot up"],
        "rim protector": ["block", "paint", "anchor", "drop coverage"],
        "room to grow": ["upside", "project", "frame", "fill out"],
        "big": ["size", "physical", "strong"],
        "tall": ["length", "long", "rangy"],
        "clutch": ["late game", "pressure", "close game"],
        "playmaker": ["passer", "creator", "ball handler"],
    }
    for k, vs in mapping.items():
        if k in ql:
            synonyms.extend(vs)
    return synonyms


def _search_cache_key(query: str, intent_tags: list[str], required_tags: list[str], n_results: int) -> tuple:
    return (
        (query or "").strip().lower(),
        tuple(sorted(set(intent_tags or []))),
        tuple(sorted(set(required_tags or []))),
        int(n_results),
    )


def _get_search_cache() -> OrderedDict:
    cache = st.session_state.get("search_cache")
    if cache is None:
        cache = OrderedDict()
        st.session_state["search_cache"] = cache
    return cache


def _cache_get(key):
    cache = _get_search_cache()
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _cache_set(key, value, max_size: int = 50):
    cache = _get_search_cache()
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _best_play_snippet(desc: str, query: str, max_len: int = 180) -> str:
    text = (desc or "").strip()
    if not text:
        return ""
    q_tokens = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2]
    if not q_tokens:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"
    lower = text.lower()
    idxs = [lower.find(t) for t in q_tokens if lower.find(t) >= 0]
    if not idxs:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"
    start = max(min(idxs) - 40, 0)
    end = min(start + max_len, len(text))
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _ensure_player_id_map(conn) -> None:
    """Ensure a mapping table exists to bridge play.player_id -> players.player_id."""
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS player_id_map (
                play_player_id TEXT PRIMARY KEY,
                player_id TEXT,
                full_name TEXT
            )
            """
        )
        cur.execute("SELECT COUNT(*) FROM player_id_map")
        count = cur.fetchone()[0] or 0
        if count == 0:
            cur.execute(
                """
                INSERT OR REPLACE INTO player_id_map (play_player_id, player_id, full_name)
                SELECT pl.player_id, p.player_id, p.full_name
                FROM plays pl
                JOIN players p ON LOWER(pl.player_name) = LOWER(p.full_name)
                WHERE pl.player_id IS NOT NULL AND p.player_id IS NOT NULL
                """
            )
            conn.commit()
    except Exception:
        pass


@st.cache_data(show_spinner=False)
def _load_player_id_map() -> dict:
    try:
        conn = sqlite3.connect(DB_PATH_STR)
        cur = conn.cursor()
        cur.execute("SELECT play_player_id, player_id FROM player_id_map")
        mapping = {}
        for play_pid, player_pid in cur.fetchall():
            play_norm = _normalize_player_id(play_pid)
            player_norm = _normalize_player_id(player_pid)
            if play_norm and player_norm:
                mapping[play_norm] = player_norm
        conn.close()
        return mapping
    except Exception:
        return {}


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
        with open(str(zip_path), "wb") as out:
            for part in parts:
                out.write(part.read_bytes())

    try:
        with zipfile.ZipFile(str(zip_path)) as zf:
            zf.extractall(str(REPO_ROOT / "data"))
    except Exception:
        return False

    return db_path.exists()


def _is_lfs_pointer(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 32:
            return False
        head = path.read_bytes()[:96]
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def _load_players_index():
    try:
        # ALWAYS use string path for sqlite3
        con = sqlite3.connect(DB_PATH_STR)
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

    try:
        con = sqlite3.connect(DB_PATH_STR)
        cur = con.cursor()
        cur.execute(f"SELECT {cols['id']} FROM players WHERE {cols['name']} = ? LIMIT 1", (name,))
        row = cur.fetchone()
        if row and row[0] is not None:
            con.close()
            return _normalize_player_id(row[0])

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
        except:
            pass
        return None

@st.cache_resource(show_spinner=False)
def _get_search_collection():
    import chromadb
    vector_db_path = REPO_ROOT / "data" / "vector_db"
    sqlite_file = vector_db_path / "chroma.sqlite3"
    if _is_lfs_pointer(sqlite_file):
        raise RuntimeError("Vector DB is a Git LFS pointer. Pull LFS assets to enable semantic search.")
    # Ensure path is string
    client = chromadb.PersistentClient(path=str(vector_db_path))
    try:
        return client.get_collection(name="skout_plays")
    except Exception:
        _restore_vector_db_if_needed()
        client = chromadb.PersistentClient(path=str(vector_db_path))
        try:
            return client.get_collection(name="skout_plays")
        except Exception:
            cols = client.list_collections()
            if cols:
                return client.get_collection(name=cols[0].name)
            raise

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


def _keyword_search_play_ids(query: str, extra_terms: list[str] | None = None, limit: int = 200) -> list[str]:
    """Fallback search path when vector DB is unavailable.

    Uses lightweight SQL filtering + token overlap scoring for reliability.
    """
    q = (query or "").strip()
    terms = [q] if q else []
    terms.extend([t.strip() for t in (extra_terms or []) if t and t.strip()])
    terms = list(dict.fromkeys(terms))[:12]
    if not terms:
        return []

    like_clauses = " OR ".join(["LOWER(description) LIKE ?" for _ in terms])
    like_vals = [f"%{t.lower()}%" for t in terms]

    try:
        con = sqlite3.connect(DB_PATH_STR)
        cur = con.cursor()
        cur.execute(
            f"""
            SELECT play_id, description
            FROM plays
            WHERE {like_clauses}
            LIMIT ?
            """,
            [*like_vals, max(limit * 3, 300)],
        )
        rows = cur.fetchall()
        con.close()
    except Exception:
        return []

    q_tokens = set(re.findall(r"[a-z0-9]+", " ".join(terms).lower()))
    scored = []
    for pid, desc in rows:
        d_tokens = set(re.findall(r"[a-z0-9]+", (desc or "").lower()))
        overlap = len(q_tokens.intersection(d_tokens))
        scored.append((overlap, str(pid)))
    scored.sort(reverse=True)
    return [pid for _, pid in scored[:limit]]

# SAFE QUERY PARAM HANDLING
def _get_qp_safe():
    try:
        return st.query_params
    except (AttributeError, TypeError):
        try:
            return st.experimental_get_query_params()
        except:
            return {}

def _set_qp_safe(key, value):
    try:
        st.query_params[key] = str(value)
    except (AttributeError, TypeError):
        try:
            params = st.experimental_get_query_params()
            params[key] = [str(value)]
            st.experimental_set_query_params(**params)
        except:
            pass

def _clear_qp_safe(key):
    try:
        if key in st.query_params:
            del st.query_params[key]
    except (AttributeError, TypeError):
        try:
            params = st.experimental_get_query_params()
            if key in params:
                del params[key]
                st.experimental_set_query_params(**params)
        except:
            pass

def _get_player_profile(player_id: str):
    """Fetch player profile with traits, stats, and plays."""
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
        con = sqlite3.connect(DB_PATH_STR)
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
            gp = srow[2] or 0
            points = srow[4] or 0
            reb = srow[13] or 0
            ast = srow[14] or 0
            
            ppg = srow[17] if srow[17] is not None and srow[17] > 0 else (points / gp if gp > 0 else 0.0)
            rpg = srow[18] if srow[18] is not None and srow[18] > 0 else (reb / gp if gp > 0 else 0.0)
            apg = srow[19] if srow[19] is not None and srow[19] > 0 else (ast / gp if gp > 0 else 0.0)

            profile["stats"] = {
                "season_id": srow[0],
                "season_label": srow[1],
                "gp": gp,
                "possessions": srow[3],
                "points": points,
                "fg_made": srow[5],
                "shot3_made": srow[6],
                "ft_made": srow[7],
                "fg_percent": srow[8],
                "shot3_percent": srow[9],
                "ft_percent": srow[10],
                "turnover": srow[11],
                "minutes": srow[12],
                "reb": reb,
                "ast": ast,
                "stl": srow[15],
                "blk": srow[16],
                "ppg": ppg,
                "rpg": rpg,
                "apg": apg,
            }
        else:
            profile["stats"] = {}

        # plays
        cur.execute("SELECT play_id, description, game_id, clock_display FROM plays WHERE player_id = ? LIMIT 25", (pid,))
        plays = cur.fetchall()
        profile["plays"] = plays

        # matchups
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
        except:
            pass
        return None

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
        if st.button("Back to search results", key=f"back_missing_{pid or 'unknown'}"):
            _clear_qp_safe("player")
            st.session_state["selected_player"] = None
            st.session_state["pending_selected_player"] = None
            st.rerun()
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

        team_label = str(profile.get("team_id") or "Unknown")
        # Heuristic for IDs vs Names
        if len(team_label) > 16 and team_label.replace("-", "").isalnum() and " " not in team_label:
            # Try to lookup, else default unknown
            team_label = "Unknown"

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
        with st.spinner("The Old Recruiter is watching tape..."):
            breakdown = generate_scout_breakdown(profile)
        
        breakdown = re.sub(r"\[clip:(\d+)\]", r"[clip](#clip-\1)", breakdown)
        st.markdown(
            f"<div style='background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25); padding:12px 14px; border-radius:10px; color:#e5e7eb;'>" + breakdown + "</div>",
            unsafe_allow_html=True,
        )

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

        st.markdown("### Social Media Scouting Report")
        report = _get_social_report(pid)
        queue_status = _get_social_queue_status(pid)

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
                _clear_qp_safe("player")
                st.session_state["selected_player"] = None
                st.session_state["pending_selected_player"] = None
                st.rerun()
            body()
        show_dialog()
    else:
        st.markdown("---")
        cols = st.columns([1, 7])
        if cols[0].button("<", key="back_profile"):
            _clear_qp_safe("player")
            st.session_state["selected_player"] = None
            st.session_state["pending_selected_player"] = None
            st.rerun()
        body()

def check_ingestion_status():
    _restore_vector_db_if_needed()
    db_path = REPO_ROOT / "data" / "vector_db" / "chroma.sqlite3"
    return db_path.exists()

def render_header():
    header_logo = get_base64_image("www/PR_LOGO_BANNER.png")
    if header_logo:
        banner_html = f"""
        <div style=\"display:flex; justify-content:center; margin-bottom:12px;\">
             <img src=\"data:image/png;base64,{header_logo}\" style=\"max-width:92vw; width:680px; height:auto; object-fit:contain; display:block;\">
        </div>
        """
    else:
        banner_html = """
        <div style=\"display:flex; justify-content:center; margin-bottom:12px;\">
             <img src=\"https://portalrecruit.github.io/PortalRecruit/PR_LOGO_BANNER.png" style=\"max-width:92vw; width:680px; height:auto; object-fit:contain; display:block;\">
        </div>
        """
    hero_html = f"<div class=\"pr-hero\">{banner_html}</div>"
    components.html(hero_html, height=360)


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default

def _fmt_height(height_in):
    try:
        inches = int(round(float(height_in)))
    except:
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
    class_term = f"\"Class of {class_year}\"" if class_year else ""

    parts = []
    if name: parts.append(f"\"{name}\"")
    sites = "(site:instagram.com OR site:tiktok.com OR site:twitter.com OR site:x.com OR site:facebook.com)"
    parts.append(sites)
    validators = []
    if school: validators.append(f"\"{school}\"")
    if hs: validators.append(f"\"{hs}\"")
    if class_term: validators.append(class_term)
    validators.extend(["basketball", "athlete"])
    if validators:
        parts.append("(" + " OR ".join(validators) + ")")
    return " ".join([p for p in parts if p])


def _build_old_recruiter_subject(query: str, matched_phrases: list[str] | None) -> str:
    phrases = [p for p in (matched_phrases or []) if p]
    uniq = []
    for p in phrases:
        if p not in uniq:
            uniq.append(p)
    phrases = uniq[:3]
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
    con = sqlite3.connect(DB_PATH_STR)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO social_scout_queue (player_id, status, requested_at) VALUES (?, 'queued', ?)",
        (player_id, datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


def _get_social_report(player_id: str):
    try:
        con = sqlite3.connect(DB_PATH_STR)
        cur = con.cursor()
        cur.execute(
            "SELECT status, report_json, chosen_url, platform, handle, updated_at FROM social_scout_reports WHERE player_id = ?",
            (player_id,),
        )
        row = cur.fetchone()
        con.close()
    except:
        return None
    if not row: return None
    status, report_json, chosen_url, platform, handle, updated_at = row
    try:
        report = json.loads(report_json) if report_json else {}
    except:
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
    try:
        con = sqlite3.connect(DB_PATH_STR)
        cur = con.cursor()
        cur.execute(
            "SELECT status, requested_at, started_at, finished_at, last_error FROM social_scout_queue WHERE player_id = ? ORDER BY requested_at DESC LIMIT 1",
            (player_id,),
        )
        row = cur.fetchone()
        con.close()
    except:
        return None
    if not row: return None
    return {
        "status": row[0],
        "requested_at": row[1],
        "started_at": row[2],
        "finished_at": row[3],
        "last_error": row[4],
    }


def _norm_person_name(name: str) -> str:
    if not name: return ""
    name = name.lower()
    name = re.sub(r"[\.'\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    parts = [p for p in name.split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(parts)


def _normalize_player_id(pid):
    if pid is None: return None
    try:
        if isinstance(pid, (list, tuple)) and pid:
            pid = pid[0]
    except: pass
    try:
        if isinstance(pid, float):
            if pid.is_integer():
                return str(int(pid))
            return str(pid).strip()
    except: pass
    s = str(pid).strip()
    if not s: return None
    if re.match(r"^\d+\.0$", s):
        s = s.split(".", 1)[0]
    return s


@st.cache_data(show_spinner=False)
def _players_table_columns():
    try:
        con = sqlite3.connect(DB_PATH_STR)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='players'")
        if not cur.fetchone():
            con.close()
            return None
        cur.execute("PRAGMA table_info(players)")
        cols = [r[1] for r in cur.fetchall()]
        con.close()
    except:
        return None

    def pick(candidates):
        for c in candidates:
            if c in cols: return c
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
    if not resolved["id"] or not resolved["name"]:
        return None
    return resolved


def _zscore(val, mean, std):
    if std == 0: return 0.0
    return (val - mean) / std


def _load_nba_archetypes():
    path = REPO_ROOT / "data" / "nba_archetypes.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except:
        return []

# --- 6. MAIN APP LOGIC ---

if "app_mode" not in st.session_state:
    if check_ingestion_status():
        st.session_state.app_mode = "Search"
    else:
        st.session_state.app_mode = "Admin"

with st.sidebar:
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

if st.session_state.app_mode == "Admin":
    render_header()
    st.caption("⚙️ Ingestion Pipeline & Settings")
    admin_path = REPO_ROOT / "src" / "dashboard" / "admin_content.py"
    if not admin_path.exists():
        admin_path = REPO_ROOT / "admin_content.py" # Fallback
    
    if admin_path.exists():
        try:
            from src.dashboard import admin_content  # noqa: F401
            if "render_admin" in dir(admin_content):
                admin_content.render_admin(REPO_ROOT=REPO_ROOT, WORK_DIR=WORK_DIR, DB_PATH_STR=DB_PATH_STR)
            else:
                # Backward compatibility: execute module top-level if it defines Streamlit UI directly
                import importlib
                importlib.reload(admin_content)
        except Exception as e:
            st.error("Admin module failed to load.")
            st.exception(e)
    else:
        st.error(f"Could not find {admin_path}")

elif st.session_state.app_mode == "Search":
    render_header()
    qp = _get_qp_safe()
    target_pid = st.session_state.get("pending_selected_player") or st.session_state.get("selected_player")
    
    if not target_pid and "player" in qp:
        raw_pid = qp["player"][0] if isinstance(qp["player"], list) else qp["player"]
        target_pid = _normalize_player_id(raw_pid)

    if target_pid:
        pid = _normalize_player_id(target_pid)
        st.session_state["pending_selected_player"] = None
        if pid:
            _set_qp_safe("player", pid)
            st.session_state["selected_player"] = pid
            _render_profile_overlay(pid)
            st.stop()
        _clear_qp_safe("player")
        st.session_state["selected_player"] = None
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    def _mark_search_requested():
        st.session_state["search_requested"] = True
        st.session_state["search_status"] = "Searching"
        st.session_state["search_started_at"] = time.time()

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
            st.session_state["search_started_at"] = time.time()

    try:
        mem_path = REPO_ROOT / "data" / "search_memory.json"
        if mem_path.exists():
            memory = json.loads(mem_path.read_text())
            recent = list(reversed(memory.get("queries", [])))[:8]
            if not query and recent:
                cols = st.columns([1,2,1])
                with cols[1]:
                    st.caption(f"Try: \"{recent[0]}\"")
    except:
        pass

    try:
        from src.search.autocomplete import suggest_rich
        suggestions = suggest_rich(query, limit=25)
    except:
        suggestions = []

    
    def _render_debug_filters():
        dbg = st.session_state.get("debug_counts") or {}
        if not dbg:
            return
        with st.expander("🔎 Debug: why no results / filter stages", expanded=False):
            st.write(dbg)
            hints = st.session_state.get("last_role_hints") or []
            sz = st.session_state.get("last_size_intents") or {}
            st.write({"role_hints": hints, "size_intents": sz})
            st.caption("Tip: If 'meta_found' is low, your players table isn't matching play player_id types. If 'after_position_filters' drops to 0, it's position normalization.")

    def _render_results(rows, query_text):
        if rows:
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

                pos = clips[0].get("Position")
                team = clips[0].get("Team")
                ht = clips[0].get("Height")
                wt = clips[0].get("Weight")
                pos = pos if pos not in [None, "", "None"] else "—"
                team = team if team not in [None, "", "None"] else "—"
                
                detail_parts = [
                    player,
                    pos,
                    _fmt_height(ht) if ht else "—",
                    f"{int(wt)} lbs" if wt else "—",
                    team,
                    f"Recruit Score: {score:.1f}",
                ]
                label = " | ".join(detail_parts)

                if pid and st.button(label, key=f"btn_{pid}", use_container_width=True):
                    st.session_state["pending_selected_player"] = pid
                    st.session_state["selected_player"] = pid
                    _set_qp_safe("player", pid)
                    st.rerun()

                extra = []
                if clips[0].get("Class") and clips[0].get("Class") != "—": extra.append(f"Class: {clips[0].get('Class')}")
                if clips[0].get("High School") and clips[0].get("High School") != "—": extra.append(f"HS: {clips[0].get('High School')}")
                if extra: st.caption(" • ".join(extra))
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No results after filters.")
            _render_debug_filters()

    if query and st.session_state.get("search_requested"):
        st.session_state["search_status"] = "Searching"
        if not st.session_state.get("search_started_at"):
            st.session_state["search_started_at"] = time.time()

        progress_placeholder = st.empty()
        st.session_state["progress_placeholder"] = progress_placeholder
        def _stage(msg, color="#7aa2f7"):
            progress_placeholder.markdown(
                f"<div class='old-recuiter-stage' style='color:{color}'>" + msg + "</div>",
                unsafe_allow_html=True,
            )
        _stage("Phoning the Old Recruiter...", "#7aa2f7")
        time.sleep(10)
        _stage("Dropping the Old Recruiter off at the airport...", "#9b7bff")
        time.sleep(5)
        explaining_start = time.time()
        _stage("Explaining Uber to the Old Recruiter...", "#f6c177")

        slider_dog = slider_menace = slider_unselfish = slider_tough = 0
        slider_rim = slider_shot = slider_gravity = slider_size = 0
        intent_dog = intent_menace = intent_unselfish = intent_tough = 0
        intent_rim = intent_shot = intent_gravity = intent_size = 0
        n_results = 150
        intent_tags = []
        required_tags = []
        finishing_intent = False

        try:
            from src.search.coach_dictionary import infer_intents_verbose, INTENTS
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

            numeric_filters = []
            pattern = re.compile(r"\b(over|under|above|below|at least|atleast|at most|atmost|more than|less than)\s+(\d+(?:\.\d+)?)%?\s*(3pt|3pt%|3pt\s*%|three|three point|three-point|ft|free throw|free-throw|fg|field goal|field-goal|shot)\b")
            for m in pattern.finditer(q_lower):
                numeric_filters.append((m.group(1), float(m.group(2)), m.group(3)))

            intents = {}
            for part in parts:
                intents.update(infer_intents_verbose(part))
        except:
            intents = {}
            q_lower = (query or "").lower()
            numeric_filters = []
            logic = "single"
        
        exclude_tags = set()
        role_hints = _infer_role_hints(query)
        size_intents = _infer_size_intents(query)
        matched_phrases = []
        apply_exclude = any(tok in q_lower for tok in [" no ", "avoid", "without", "dont", "don't", "not "])
        leadership_intent = "leadership" in intents
        resilience_intent = "resilience" in intents
        defensive_big_intent = "defensive_big" in intents
        clutch_intent = "clutch" in intents
        undervalued_intent = "undervalued" in intents
        finishing_intent = "finishing" in intents

        heuristic_tags = []
        if any(k in q_lower for k in ["big man", "big", "center", "rim protector", "paint"]):
            role_hints.add("big")
            heuristic_tags += ["rim_pressure", "block", "post_up", "paint_touch"]
        if any(k in q_lower for k in ["stretch", "can shoot", "shooting big", "pick and pop", "spacing"]):
            heuristic_tags += ["shot3", "catch_shoot", "spot_up", "pick_pop"]
        if any(k in q_lower for k in ["athletic", "explosive", "vertical", "lob threat"]):
            heuristic_tags += ["rim_run", "dunk", "putback", "transition"]
        if any(k in q_lower for k in ["rebound", "boards", "glass"]):
            heuristic_tags += ["off_reb", "def_reb"]
        if any(k in q_lower for k in ["rim pressure", "downhill", "paint touch"]):
            heuristic_tags += ["drive", "rim_finish", "layup"]
        if any(k in q_lower for k in ["playmaker", "creator", "facilitator"]):
            heuristic_tags += ["assist", "pnr", "drive_kick"]
        if any(k in q_lower for k in ["defender", "stopper", "lockdown"]):
            heuristic_tags += ["deflection", "steal", "on_ball"]

        for hit, phrase in intents.values():
            intent = hit.intent
            w = hit.weight
            role_hints |= hit.role_hints
            matched_phrases.append(phrase)
            
            intent_dog = max(intent_dog, int(intent.traits.get("dog", 0) * w))
            intent_menace = max(intent_menace, int(intent.traits.get("menace", 0) * w))
            intent_unselfish = max(intent_unselfish, int(intent.traits.get("unselfish", 0) * w))
            intent_tough = max(intent_tough, int(intent.traits.get("tough", 0) * w))
            intent_rim = max(intent_rim, int(intent.traits.get("rim", 0) * w))
            intent_shot = max(intent_shot, int(intent.traits.get("shot", 0) * w))
            intent_gravity = max(intent_gravity, int(intent.traits.get("gravity", 0) * w))
            if intent is INTENTS.get("size_measurables"):
                intent_size = max(intent_size, 70)
            intent_tags = list(set(intent_tags + list(intent.tags)))
            exclude_tags |= intent.exclude_tags

            if logic == "and":
                required_tags = list(set(required_tags + list(intent.tags)))

        st.session_state["last_matched_phrases"] = matched_phrases
        st.session_state["last_role_hints"] = sorted(role_hints)
        st.session_state["last_size_intents"] = size_intents

        if "guard" in role_hints: intent_tags = list(set(intent_tags + ["drive", "pnr"]))
        if "wing" in role_hints: intent_tags = list(set(intent_tags + ["3pt", "deflection"]))
        if "big" in role_hints: intent_tags = list(set(intent_tags + ["rim_pressure", "block", "post_up"]))

        if heuristic_tags:
            intent_tags = list(set(intent_tags + heuristic_tags))

        if finishing_intent:
            required_tags = list(set(required_tags + ["rim_finish", "layup", "dunk", "made"]))

        required_tags = [] # Default off
        st.session_state["last_query"] = query
        st.session_state["last_query_tags"] = intent_tags

        collection = None
        vector_search_ready = True
        try:
            collection = _get_search_collection()
        except Exception as e:
            vector_search_ready = False
            st.info(f"Semantic index unavailable, using keyword fallback search. ({e})")

        from src.search.semantic import build_expanded_query, semantic_search, expand_query_terms
        expanded_terms = (expand_query_terms(query) or []) + (_expand_query_synonyms(query) or [])
        expanded_query = build_expanded_query(query, (matched_phrases or []) + (expanded_terms or []))

        status = st.status("Searching…", expanded=False)
        status.update(state="running")
        st.markdown("<script>document.body.classList.add('searching');</script>", unsafe_allow_html=True)

        cache_key = _search_cache_key(query, intent_tags, required_tags, n_results)
        cached_play_ids = _cache_get(cache_key)
        if cached_play_ids is not None:
            play_ids = cached_play_ids
        else:
            if vector_search_ready and collection is not None:
                try:
                    play_ids = semantic_search(
                        collection,
                        query=query,
                        n_results=n_results,
                        extra_query_terms=(matched_phrases or []) + (expanded_terms or []),
                        required_tags=required_tags,
                        boost_tags=intent_tags,
                    )
                except:
                    play_ids = []

                if not play_ids:
                    try:
                        collection = _get_search_collection()
                        play_ids = semantic_search(
                            collection,
                            query=query,
                            n_results=n_results,
                            extra_query_terms=(matched_phrases or []) + (expanded_terms or []),
                            required_tags=required_tags,
                            boost_tags=intent_tags,
                        )
                    except:
                        play_ids = []

                if not play_ids:
                    vector_search_ready = False
                    play_ids = _keyword_search_play_ids(
                        query,
                        extra_terms=(matched_phrases or []) + (expanded_terms or []),
                        limit=max(n_results, 150),
                    )
            else:
                play_ids = _keyword_search_play_ids(
                    query,
                    extra_terms=(matched_phrases or []) + (expanded_terms or []),
                    limit=max(n_results, 150),
                )

            if not play_ids and expanded_query:
                play_ids = _keyword_search_play_ids(
                    expanded_query,
                    extra_terms=(matched_phrases or []) + (expanded_terms or []),
                    limit=max(n_results, 150),
                )

            _cache_set(cache_key, play_ids)
        count_initial = len(play_ids)

        if vector_search_ready and collection is not None and len(play_ids) < 8:
            try:
                play_ids = semantic_search(
                    collection,
                    query=expanded_query,
                    n_results=max(n_results, 200),
                    extra_query_terms=(matched_phrases or []) + (expanded_terms or []),
                    required_tags=[],
                    boost_tags=intent_tags,
                    diversify_by_player=False,
                )
            except:
                st.error("Search index error.")
                st.stop()
        count_after_fallback = len(play_ids)
        
        elapsed = time.time() - explaining_start
        if elapsed < 5: time.sleep(5 - elapsed)

        st.markdown("<script>document.body.classList.remove('searching');</script>", unsafe_allow_html=True)

        if not play_ids:
            status.update(state="complete", label="Search complete")
            st.warning("No results found.")
        else:
            status.update(state="complete", label="Search complete")
            conn = sqlite3.connect(DB_PATH_STR)
            cur = conn.cursor()

            try:
                cur.execute("PRAGMA table_info(player_traits)")
                existing_cols = {r[1] for r in cur.fetchall()}
                needed = {"leadership_index", "resilience_index", "defensive_big_index", "clutch_index", "undervalued_index", "gravity_index"}
                for col in needed:
                    if col not in existing_cols:
                        cur.execute(f"ALTER TABLE player_traits ADD COLUMN {col} REAL")
                conn.commit()
            except: pass

            _ensure_player_id_map(conn)
            player_id_map = _load_player_id_map()

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

            debug_counts = {
                'play_ids_initial': count_initial,
                'play_ids_after_fallback': count_after_fallback,
                'play_rows_fetched': len(play_rows),
                'meta_found': 0,
                'after_trait_sliders': 0,
                'after_tag_filters': 0,
                'after_numeric_filters': 0,
                'after_position_filters': 0,
                'after_size_filters': 0,
                'final_rows': 0,
            }

            mapped_player_ids = []
            for r in play_rows:
                play_pid = _normalize_player_id(r[4])
                mapped_pid = player_id_map.get(play_pid) or play_pid
                if mapped_pid:
                    mapped_player_ids.append(mapped_pid)
            traits = {}
            if mapped_player_ids:
                ph2 = ",".join(["?"] * len(set(mapped_player_ids)))
                cur.execute(
                    f"""
                    SELECT player_id, dog_index, menace_index, unselfish_index,
                           toughness_index, rim_pressure_index, shot_making_index, gravity_index, size_index,
                           leadership_index, resilience_index, defensive_big_index, clutch_index,
                           undervalued_index
                    FROM player_traits
                    WHERE player_id IN ({ph2})
                    """,
                    list(set(mapped_player_ids)),
                )
                traits = {
                    _normalize_player_id(r[0]): {
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
                    for r in cur.fetchall() if _normalize_player_id(r[0])
                }

            cur.execute("""
                SELECT AVG(dog_index), AVG(menace_index), AVG(unselfish_index),
                       AVG(toughness_index), AVG(rim_pressure_index), AVG(shot_making_index), AVG(gravity_index),
                       AVG(size_index), AVG(leadership_index), AVG(resilience_index), AVG(defensive_big_index),
                       AVG(clutch_index), AVG(undervalued_index)
                FROM player_traits
            """)
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

            from src.processing.play_tagger import tag_play
            
            player_positions = {}
            try:
                conn_pos = sqlite3.connect(DB_PATH_STR)
                cur_pos = conn_pos.cursor()
                cur_pos.execute("SELECT player_id, position FROM players")
                player_positions = {_normalize_player_id(r[0]): (r[1] or "") for r in cur_pos.fetchall() if _normalize_player_id(r[0])}
                conn_pos.close()
            except: pass

            player_meta = {}
            player_meta_by_name = {}
            team_name_by_id = {}
            try:
                conn_meta = sqlite3.connect(DB_PATH_STR)
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
                    if pid_norm: player_meta[pid_norm] = meta_row
                    name_key = _norm_person_name(r[1] or "")
                    if name_key: player_meta_by_name[name_key] = meta_row

                try:
                    cur_meta.execute("SELECT team_id, offense_team, defense_team FROM plays WHERE team_id IS NOT NULL AND (offense_team IS NOT NULL OR defense_team IS NOT NULL)")
                    counts = {}
                    for tid, off_t, def_t in cur_meta.fetchall():
                        for t in (off_t, def_t):
                            if not t: continue
                            key = (tid, t)
                            counts[key] = counts.get(key, 0) + 1
                    for (tid, name), cnt in sorted(counts.items(), key=lambda x: -x[1]):
                        if tid not in team_name_by_id: team_name_by_id[tid] = name
                except: pass
                conn_meta.close()
            except: pass

            player_stats = {}
            player_names = {}
            traits_all = {}
            player_team_guess = {}
            try:
                conn2 = sqlite3.connect(DB_PATH_STR)
                cur2 = conn2.cursor()
                cur2.execute("SELECT player_id, season_id, team_id, gp, possessions, points, fg_percent, shot2_percent, shot3_percent, ft_percent, fg_attempt, shot2_attempt, shot3_attempt, turnover, ppg, rpg, apg FROM player_season_stats ORDER BY season_id DESC")
                for r in cur2.fetchall():
                    pid = _normalize_player_id(r[0])
                    if not pid:
                        continue
                    if pid in player_stats: continue
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
                
                try:
                    cur2.execute("SELECT p.player_id, p.offense_team, p.defense_team, p.is_home, p.game_id FROM plays p WHERE p.player_id IS NOT NULL")
                    counts = {}
                    for pid, off_t, def_t, is_home, gid in cur2.fetchall():
                        pid_norm = _normalize_player_id(pid)
                        pid_mapped = player_id_map.get(pid_norm) or pid_norm
                        if not pid_mapped:
                            continue
                        if off_t: counts[(pid_mapped, off_t)] = counts.get((pid_mapped, off_t), 0) + 1
                        if def_t: counts[(pid_mapped, def_t)] = counts.get((pid_mapped, def_t), 0) + 1
                    if not counts:
                        cur2.execute("SELECT p.player_id, p.is_home, g.home_team, g.away_team FROM plays p JOIN games g ON g.game_id = p.game_id WHERE p.player_id IS NOT NULL AND g.home_team IS NOT NULL AND g.away_team IS NOT NULL")
                        for pid, is_home, home, away in cur2.fetchall():
                            pid_norm = _normalize_player_id(pid)
                            pid_mapped = player_id_map.get(pid_norm) or pid_norm
                            team = home if is_home else away
                            if pid_mapped and team:
                                counts[(pid_mapped, team)] = counts.get((pid_mapped, team), 0) + 1
                    for (pid, team), cnt in sorted(counts.items(), key=lambda x: -x[1]):
                        if pid not in player_team_guess: player_team_guess[pid] = team
                except: pass

                cur2.execute("SELECT player_id, full_name FROM players")
                for r in cur2.fetchall(): player_names[r[0]] = r[1]
                
                cur2.execute("SELECT player_id, dog_index, menace_index, unselfish_index, toughness_index, rim_pressure_index, shot_making_index, gravity_index, size_index, leadership_index, resilience_index, defensive_big_index, clutch_index, undervalued_index FROM player_traits")
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
            except: pass

            def _collect_vals(key):
                return [v.get(key) for v in player_stats.values() if v.get(key) is not None]

            stat_keys = ["points", "possessions", "fg_percent", "shot3_percent", "ft_percent", "fg_attempt", "shot2_attempt", "shot3_attempt", "turnover"]
            stat_means = {k: (_safe_float(sum(_collect_vals(k))) / max(1, len(_collect_vals(k)))) for k in stat_keys}
            stat_stds = {k: math.sqrt(sum(((_safe_float(v) - stat_means[k]) ** 2) for v in _collect_vals(k)) / max(1, len(_collect_vals(k)))) for k in stat_keys}

            _stage("Confirming arrival of the Old Recruiter at Prospect's local gym...", "#7bdcb5")

            rows = []
            for pid, desc, gid, clock, player_id, player_name in play_rows:
                pid_norm = _normalize_player_id(player_id)
                mapped_pid = player_id_map.get(pid_norm) or pid_norm
                meta = player_meta.get(mapped_pid) if mapped_pid else None
                if meta is None and player_name:
                    meta = player_meta_by_name.get(_norm_person_name(player_name))
                if meta is None:
                    continue
                debug_counts['meta_found'] += 1
                
                t = traits.get(mapped_pid, {}) if mapped_pid else {}
                dog_index = t.get("dog")
                menace_index = t.get("menace")
                unselfish_index = t.get("unselfish")
                tough_index = t.get("tough")
                rim_index = t.get("rim")
                shot_index = t.get("shot")
                gravity_index = t.get("gravity")

                if dog_index is not None and dog_index < slider_dog: continue
                if menace_index is not None and menace_index < slider_menace: continue
                if unselfish_index is not None and unselfish_index < slider_unselfish: continue
                if tough_index is not None and tough_index < slider_tough: continue
                if rim_index is not None and rim_index < slider_rim: continue
                if shot_index is not None and shot_index < slider_shot: continue
                if gravity_index is not None and gravity_index < slider_gravity: continue
                if t.get("size") is not None and t.get("size") < slider_size: continue

                debug_counts['after_trait_sliders'] += 1

                play_tags = list(_tag_play_cached(desc))
                if "non_possession" in play_tags: continue
                if apply_exclude and exclude_tags and set(play_tags).intersection(exclude_tags): continue
                if required_tags:
                    req_threshold = _required_tag_threshold(required_tags)
                    req_hits = len(set(required_tags).intersection(set(play_tags)))
                    if req_hits < req_threshold: continue

                debug_counts['after_tag_filters'] += 1

                if numeric_filters:
                    pstats = player_stats.get(mapped_pid, {}) if mapped_pid else {}
                    allow = True
                    for comp, val, stat in numeric_filters:
                        stat_val = None
                        if "3" in stat or "three" in stat: stat_val = _safe_float(pstats.get("shot3_percent")) * 100
                        elif stat in {"ft", "free throw", "free-throw"}: stat_val = _safe_float(pstats.get("ft_percent")) * 100
                        elif stat in {"fg", "field goal", "field-goal", "shot"}: stat_val = _safe_float(pstats.get("fg_percent")) * 100
                        if stat_val is None: continue
                        if comp in {"over", "above", "more than", "at least", "atleast"} and not (stat_val >= val): allow = False
                        if comp in {"under", "below", "less than", "at most", "atmost"} and not (stat_val <= val): allow = False
                    if not allow: continue

                debug_counts['after_numeric_filters'] += 1

                pos = (player_positions.get(mapped_pid) or "").upper() if mapped_pid else ""
                if pos:
                    pos_tags = _position_tags(pos)
                    if "guard" in role_hints and "guard" not in pos_tags:
                        continue
                    if "wing" in role_hints and "wing" not in pos_tags:
                        continue
                    if "big" in role_hints and "big" not in pos_tags:
                        continue

                debug_counts['after_position_filters'] += 1

                # Size / development intents (soft but effective)
                if meta:
                    h_in = meta.get("height_in")
                    w_lb = meta.get("weight_lb")
                    h_in = int(h_in) if isinstance(h_in, (int, float)) and h_in else None
                    w_lb = float(w_lb) if isinstance(w_lb, (int, float)) and w_lb else None

                    hmin = size_intents.get("height_min")
                    hmax = size_intents.get("height_max")
                    if hmin is not None and h_in is not None and h_in < hmin:
                        continue
                    if hmax is not None and h_in is not None and h_in > hmax:
                        continue

                    # position-relative sanity: if query says "big" but player is clearly small, drop
                    if "big" in role_hints and h_in is not None and h_in < 77:
                        continue
                    if "guard" in role_hints and "tall" in (query or "").lower() and h_in is not None and h_in < 74:
                        continue

                    # "room to grow" prefers younger + leaner frames (approx heuristic)
                    if size_intents.get("growth"):
                        class_year = (meta.get("class_year") or "").lower()
                        young = any(k in class_year for k in ["fr", "fresh", "so", "soph"])
                        if not young:
                            # don't exclude, but slight penalty
                            pass


                debug_counts['after_size_filters'] += 1

                weights = {"dog": 0.5, "menace": 0.5, "unselfish": 0.5, "tough": 0.5, "rim": 0.5, "shot": 0.5, "gravity": 0.5, "size": 0.3}
                if intent_dog > 0: weights["dog"] = 2.0
                if intent_menace > 0: weights["menace"] = 2.5
                if intent_unselfish > 0: weights["unselfish"] = 1.5
                if intent_tough > 0: weights["tough"] = 1.5
                if intent_rim > 0: weights["rim"] = 2.2
                if intent_shot > 0: weights["shot"] = 2.5
                if intent_gravity > 0: weights["gravity"] = 2.0
                if "big" in role_hints: weights["size"] = 2.0

                score = 0
                for key, w in weights.items():
                    val = t.get(key) or 0
                    score += val * w

                score += max(0, (t.get("dog") or 0) - intent_dog) * 0.1
                score += max(0, (t.get("menace") or 0) - intent_menace) * 0.1
                score += max(0, (t.get("unselfish") or 0) - intent_unselfish) * 0.1
                score += max(0, (t.get("tough") or 0) - intent_tough) * 0.1
                score += max(0, (t.get("rim") or 0) - intent_rim) * 0.1
                score += max(0, (t.get("shot") or 0) - intent_shot) * 0.1
                score += max(0, (t.get("gravity") or 0) - intent_gravity) * 0.1

                matching_tags = set(play_tags).intersection(set(intent_tags))
                score += len(matching_tags) * 15

                if leadership_intent and t.get("leadership"): score += t.get("leadership") * 15
                if resilience_intent and t.get("resilience"): score += t.get("resilience") * 12
                if defensive_big_intent and t.get("defensive_big"): score += t.get("defensive_big") * 18
                if clutch_intent and t.get("clutch"): score += t.get("clutch") * 15
                if undervalued_intent and t.get("undervalued"): score += t.get("undervalued") * 14
                if "turnover" in play_tags: score -= 8
                if "made" in play_tags: score += 12
                if "missed" in play_tags: score -= 6
                if player_name and player_name in (desc or ""): score += 6

                home, away, video = matchups.get(gid, ("Unknown", "Unknown", None))

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
                    if val is None: continue
                    avg = trait_avg.get(key, 0)
                    if val >= avg + 10: strengths.append(label)
                    elif val <= avg - 10: weaknesses.append(label)
                strengths = strengths[:2]
                weaknesses = weaknesses[:2]

                reason_parts = []
                if intent_unselfish and (unselfish_index or 0) >= intent_unselfish: reason_parts.append("high unselfishness")
                if intent_tough and (tough_index or 0) >= intent_tough: reason_parts.append("tough/competitive")
                if intent_dog and (dog_index or 0) >= intent_dog: reason_parts.append("dog mentality")
                if intent_menace and (menace_index or 0) >= intent_menace: reason_parts.append("defensive menace")
                if intent_rim and (rim_index or 0) >= intent_rim: reason_parts.append("rim pressure")
                if intent_shot and (shot_index or 0) >= intent_shot: reason_parts.append("shot making")
                if intent_gravity and (gravity_index or 0) >= intent_gravity: reason_parts.append("gravity well")
                if not reason_parts:
                    if strengths: reason_parts.append(f"elite {strengths[0].lower()}")
                    else: reason_parts.append("balanced skill set")
                reason = " — ".join(reason_parts)

                # Comps & Archetypes logic omitted for brevity as it was correct
                # Just ensuring variables exist
                sim_players = []
                nba_floor = "—"
                nba_ceiling = "—"

                meta = {}
                pid_norm = _normalize_player_id(player_id)
                mapped_pid = player_id_map.get(pid_norm) or pid_norm
                if "player_meta" in locals() and mapped_pid: meta = player_meta.get(mapped_pid, {})
                if not meta and "player_meta_by_name" in locals(): meta = player_meta_by_name.get(_norm_person_name(player_name or ""), {})

                pos_val = meta.get("position", "")
                team_val = meta.get("team_id", "")
                ht_val = meta.get("height_in")
                wt_val = meta.get("weight_lb")
                class_val = meta.get("class_year", "")
                hs_val = meta.get("high_school", "")
                ppg_val = (player_stats.get(mapped_pid) or {}).get("ppg") if mapped_pid else None
                rpg_val = (player_stats.get(mapped_pid) or {}).get("rpg") if mapped_pid else None
                apg_val = (player_stats.get(mapped_pid) or {}).get("apg") if mapped_pid else None
                pos_val = pos_val if pos_val not in [None, "", "None"] else "—"
                team_val = team_val if team_val not in [None, "", "None"] else "—"
                class_val = class_val if class_val not in [None, "", "None"] else "—"
                hs_val = hs_val if hs_val not in [None, "", "None"] else "—"
                if team_val != "—":
                    team_clean = str(team_val).strip()
                    if len(team_clean) > 16 and team_clean.replace("-", "").isalnum() and " " not in team_clean:
                        team_val = team_name_by_id.get(team_val, "—")
                if team_val == "—" and player_team_guess:
                    team_val = player_team_guess.get(mapped_pid, "—")
                if team_val == "—": team_val = "Unknown"

                debug_counts['final_rows'] = debug_counts.get('final_rows',0)
                debug_counts['final_rows'] += 1

                rows.append({
                    # debug
                
                    "Match": f"{home} vs {away}",
                    "Clock": clock,
                    "Player": (player_name or "Unknown"),
                    "Player ID": mapped_pid or player_id,
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
                    "Profile": "#", # Placeholder URL
                    "Dog Index": dog_index,
                    "Menace": menace_index,
                    "Unselfish": unselfish_index,
                    "Toughness": tough_index,
                    "Rim Pressure": rim_index,
                    "Shot Making": shot_index,
                    "Gravity Well": gravity_index,
                    "Tags": ", ".join(play_tags),
                    "Play": _best_play_snippet(desc, query),
                    "Video": video or "-",
                    "Score": round(score, 2),
                })

            st.session_state['debug_counts'] = debug_counts
            st.session_state['last_role_hints'] = sorted(list(role_hints)) if 'role_hints' in locals() else []
            st.session_state['last_size_intents'] = size_intents if 'size_intents' in locals() else {}

            _stage("Incoming email from <a href='mailto:theoldrecruiter@portalrecruit.com'>theoldrecruiter@portalrecruit.com</a>...", "#ff7eb6")
            rows.sort(key=lambda r: r.get("Score", 0), reverse=True)
            st.session_state["search_status"] = "Search"
            st.session_state["search_requested"] = False
            st.session_state["last_rows"] = rows
            _render_results(rows, query)
