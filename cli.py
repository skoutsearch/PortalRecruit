#!/home/jch903/.venv_310/bin/python
#!/home/jch903/.venv_310/bin/python
import argparse
import os
import random
import sqlite3
from typing import Any, Dict, List, Optional

import chromadb

from src.search.semantic import semantic_search, _lexical_overlap_score, _tokenize
from src.llm.scout import generate_scout_breakdown

# ANSI colors
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_BLUE_BOLD = "\033[1;34m"
ANSI_RESET = "\033[0m"

VECTOR_DB_PATH = os.path.join(os.getcwd(), "data/vector_db")
DB_PATH = os.path.join(os.getcwd(), "data/skout.db")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except Exception:
        pass
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass


def _get_player_profile(conn, player_id: Optional[str], player_name: Optional[str], original_desc: Optional[str] = None) -> Dict[str, Any]:
    cur = conn.cursor()
    mapped_pid = None
    if player_id:
        try:
            cur.execute("SELECT player_id FROM player_id_map WHERE play_player_id = ?", (player_id,))
            row = cur.fetchone()
            if row and row[0]:
                mapped_pid = row[0]
        except Exception:
            mapped_pid = None
    pid = mapped_pid or player_id

    profile = {
        "player_id": pid,
        "name": player_name or "Unknown",
        "team_id": None,
        "height_in": None,
        "weight_lb": None,
        "class_year": None,
        "traits": {},
        "stats": {},
        "plays": [],
        "original_desc": original_desc,
    }

    if pid:
        cur.execute(
            "SELECT player_id, full_name, position, team_id, height_in, weight_lb, class_year, high_school FROM players WHERE player_id = ? LIMIT 1",
            (pid,),
        )
        row = cur.fetchone()
        if row:
            profile.update({
                "player_id": row[0],
                "name": row[1] or profile["name"],
                "position": row[2] or "",
                "team_id": row[3],
                "height_in": row[4],
                "weight_lb": row[5],
                "class_year": row[6],
                "high_school": row[7],
            })

    if player_name:
        cur.execute(
            "SELECT player_id, full_name, position, team_id, height_in, weight_lb, class_year, high_school FROM players WHERE LOWER(full_name) = LOWER(?) LIMIT 1",
            (player_name,),
        )
        row = cur.fetchone()
        if row and not profile.get("team_id"):
            profile.update({
                "player_id": row[0],
                "name": row[1] or profile["name"],
                "position": row[2] or "",
                "team_id": row[3],
                "height_in": row[4],
                "weight_lb": row[5],
                "class_year": row[6],
                "high_school": row[7],
            })
            pid = row[0]

    if pid:
        cur.execute(
            "SELECT dog_index, menace_index, unselfish_index, toughness_index, rim_pressure_index, shot_making_index, gravity_index, size_index FROM player_traits WHERE player_id = ? LIMIT 1",
            (pid,),
        )
        t = cur.fetchone()
        if t:
            profile["traits"] = {
                "dog_index": t[0],
                "menace_index": t[1],
                "unselfish_index": t[2],
                "toughness_index": t[3],
                "rim_pressure_index": t[4],
                "shot_making_index": t[5],
                "gravity_index": t[6],
                "size_index": t[7],
            }

        cur.execute(
            "SELECT season_id, season_label, gp, possessions, points, fg_made, shot3_made, ft_made, ppg, rpg, apg FROM player_season_stats WHERE player_id = ? ORDER BY season_id DESC LIMIT 1",
            (pid,),
        )
        s = cur.fetchone()
        if s:
            profile["stats"] = {
                "season_id": s[0],
                "season_label": s[1],
                "gp": s[2],
                "possessions": s[3],
                "points": s[4],
                "fg_made": s[5],
                "shot3_made": s[6],
                "ft_made": s[7],
                "ppg": s[8],
                "rpg": s[9],
                "apg": s[10],
            }

        cur.execute(
            "SELECT play_id, description, game_id, clock_display FROM plays WHERE player_id = ? ORDER BY utc DESC LIMIT 12",
            (pid,),
        )
        profile["plays"] = cur.fetchall()

    return profile


def _best_snippet(desc: str, query: str, max_len: int = 160) -> str:
    text = (desc or "").strip()
    if not text:
        return ""
    tokens = [t for t in _tokenize(query) if len(t) > 2]
    if not tokens:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"
    lower = text.lower()
    idxs = [lower.find(t) for t in tokens if lower.find(t) >= 0]
    if not idxs:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"
    start = max(min(idxs) - 30, 0)
    end = min(start + max_len, len(text))
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _colorize_outcome(text: str, tags: str) -> str:
    lower = (text or "").lower()
    t = (tags or "").lower()
    if "miss" in lower or "miss" in t or "turnover" in lower or "turnover" in t:
        return f"{ANSI_RED}{text}{ANSI_RESET}"
    if "make" in lower or "made" in t or "score" in t:
        return f"{ANSI_GREEN}{text}{ANSI_RESET}"
    return text


def _format_matchup(matchup: str, clock: str) -> str:
    parts = matchup.split(" vs ") if matchup else []
    opp = parts[1] if len(parts) == 2 else matchup
    return f"VS {opp} ({clock})" if clock else f"VS {opp}"


def run_search(query: str, n_results: int = 5, debug: bool = False, media: bool = False, biometrics: bool = False) -> None:
    _load_env()
    # TODO: Replace local Chroma call with Synergy/SportRadar search endpoint
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection(name="skout_plays")
    play_ids = semantic_search(collection, query=query, n_results=n_results)

    meta_lookup: Dict[str, Dict[str, Any]] = {}
    try:
        meta = collection.get(ids=play_ids, include=["metadatas"]) if play_ids else None
        if meta and meta.get("ids"):
            for pid, m in zip(meta.get("ids", []), meta.get("metadatas", [])):
                meta_lookup[str(pid)] = m or {}
    except Exception:
        meta_lookup = {}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"\n🔍 Search Results for: '{query}'")
    print("-" * 50)

    if not play_ids:
        print("No results found.")
        return

    query_tokens = _tokenize(query)
    results: List[Dict[str, Any]] = []

    for i, play_id in enumerate(play_ids):
        cursor.execute(
            """
            SELECT description, tags, clock_display, game_id, player_name, player_id
            FROM plays
            WHERE play_id = ?
            """,
            (play_id,),
        )
        row = cursor.fetchone()
        if not row:
            continue
        desc, tags, clock, game_id, player_name, player_id = row

        cursor.execute(
            "SELECT video_path, home_team, away_team FROM games WHERE game_id = ?",
            (game_id,),
        )
        game_row = cursor.fetchone()
        if game_row:
            v_path = os.path.basename(game_row[0]) if game_row[0] else "No Video"
            matchup = f"{game_row[1]} vs {game_row[2]}"
        else:
            v_path = "Unknown"
            matchup = "Unknown"

        meta = meta_lookup.get(str(play_id), {})
        # TODO: replace mock link with real Synergy/SportRadar video URL
        video_link = meta.get("video_url") or meta.get("url") or meta.get("s3_link")
        if not video_link:
            video_link = f"https://mock.synergy.com/video/{play_id}.mp4"

        score = _lexical_overlap_score(query_tokens, desc, {"tags": tags})
        score = min(1.0, max(0.0, score))
        snippet = _best_snippet(desc, query)

        results.append({
            "score": score,
            "player_name": player_name or "Unknown",
            "player_id": player_id,
            "desc": desc,
            "tags": tags,
            "clock": clock,
            "matchup": matchup,
            "file": v_path,
            "snippet": snippet,
            "video": video_link,
            "meta": meta,
            "original_desc": meta.get("original_desc") if meta else None,
        })

    results.sort(key=lambda r: r["score"], reverse=True)

    # normalize scores to a 0-1 range if all zeros
    if results and all((r.get("score") or 0) == 0 for r in results):
        n = len(results)
        for idx, r in enumerate(results):
            r["score"] = round((n - idx) / max(n, 1), 3)

    for i, r in enumerate(results):
        video_out = r['video'] if r.get('video') else r['file']
        name_out = f"{ANSI_BLUE_BOLD}{r['player_name']}{ANSI_RESET}"
        snippet_out = _colorize_outcome(r['snippet'], r.get('tags', ""))
        print(f"[{i+1}] Score: {r['score']:.2f} | Player: {name_out}")
        print(f" Matchup: {_format_matchup(r['matchup'], r['clock'])}")
        print(f" Snippet: {snippet_out}")
        print(f" Tags: [{r['tags']}]\n Video: {video_out}")
        print("")

    if debug and results:
        print("\n🧪 Debug Metadata (Top Result)")
        print("-" * 50)
        print(results[0].get("meta"))

    media_img = None
    if media or biometrics:
        from src.social_media import build_video_query, build_image_query, serper_search, select_best_video, select_best_image
        team_name = ""
        if results:
            team_name = results[0].get("matchup", "").split(" vs ")[0]
        v_query = build_video_query(query, team_name)
        i_query = build_image_query(query, team_name)
        v_results = serper_search(v_query, type="videos")
        i_results = serper_search(i_query, type="images")
        vid = select_best_video(v_results, query)
        media_img = select_best_image(i_results, query)
        if media:
            print("\n📺 Media Lookup")
            print("-" * 50)
            print(f"Video Query: {v_query}")
            print(f"Image Query: {i_query}")
            print(f"YouTube ID: {vid}")
            print(f"Image URL: {media_img}")

    if biometrics:
        from src.biometrics import generate_biometric_tags
        profile = _get_player_profile(conn, results[0].get("player_id") if results else None, results[0].get("player_name") if results else None)
        if profile is not None:
            profile["image_url"] = media_img
            bio = generate_biometric_tags(profile)
            print("\n🧍 Biometric Tags")
            print("-" * 50)
            print(f"Math Tags: {bio.get('math_tags')}")
            print(f"Vision: {bio.get('vision')}")
            print(f"All Tags: {bio.get('tags')}")

    top3 = results[:3] if len(results) >= 3 else results
    if not top3:
        return
    pick = random.choice(top3)
    profile = _get_player_profile(
        conn,
        pick.get("player_id"),
        pick.get("player_name"),
        original_desc=pick.get("original_desc") or pick.get("desc"),
    )
    breakdown = generate_scout_breakdown(profile)

    print("\n🏀 Scout Breakdown")
    print("-" * 50)
    print(f"Selected: {pick.get('player_name')} (score {pick.get('score'):.2f})")
    print(breakdown)


def run_interactive() -> None:
    while True:
        os.system("clear")
        q = input("Enter search query (or 'q' to quit): ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue
        run_search(q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--debug", action="store_true")
    s.add_argument("--media", action="store_true")
    s.add_argument("--biometrics", action="store_true")

    sub.add_parser("interactive")

    args = parser.parse_args()
    if args.command == "search":
        run_search(args.query, n_results=args.n, debug=args.debug, media=args.media, biometrics=args.biometrics)
    elif args.command == "interactive":
        run_interactive()
    else:
        parser.print_help()
