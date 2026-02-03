#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from pathlib import Path
import chromadb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DB_PATH = REPO_ROOT / "data" / "skout.db"
VECTOR_DB = REPO_ROOT / "data" / "vector_db"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--suggest", action="store_true", help="show autocomplete suggestions for the query")
    p.add_argument("--min_dog", type=float, default=0)
    p.add_argument("--min_menace", type=float, default=0)
    p.add_argument("--min_unselfish", type=float, default=0)
    p.add_argument("--min_tough", type=float, default=0)
    p.add_argument("--min_rim", type=float, default=0)
    p.add_argument("--min_shot", type=float, default=0)
    p.add_argument("--tags", nargs="*", default=[])
    args = p.parse_args()

    if not args.query:
        print("Provide a query, e.g. ./scripts/cli_search.py \"downhill guard\"")
        return

    if args.suggest:
        from src.search.autocomplete import suggest_rich
        print("Suggestions:", ", ".join(suggest_rich(args.query, limit=25)))
        return

    client = chromadb.PersistentClient(path=str(VECTOR_DB))
    collection = client.get_collection(name="skout_plays")
    results = collection.query(query_texts=[args.query], n_results=args.n)

    play_ids = results.get("ids", [[]])[0]
    if not play_ids:
        print("No results.")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    ph = ",".join(["?"] * len(play_ids))
    cur.execute(f"""
        SELECT play_id, description, game_id, clock_display, player_id, player_name
        FROM plays WHERE play_id IN ({ph})
    """, play_ids)
    play_rows = cur.fetchall()

    player_ids = [r[4] for r in play_rows if r[4]]
    traits = {}
    if player_ids:
        ph2 = ",".join(["?"] * len(set(player_ids)))
        cur.execute(f"""
            SELECT player_id, dog_index, menace_index, unselfish_index,
                   toughness_index, rim_pressure_index, shot_making_index
            FROM player_traits WHERE player_id IN ({ph2})
        """, list(set(player_ids)))
        traits = {
            r[0]: dict(dog=r[1], menace=r[2], unselfish=r[3], tough=r[4], rim=r[5], shot=r[6])
            for r in cur.fetchall()
        }

    game_ids = list({r[2] for r in play_rows})
    matchups = {}
    if game_ids:
        ph3 = ",".join(["?"] * len(game_ids))
        cur.execute(f"""
            SELECT game_id, home_team, away_team, video_path
            FROM games WHERE game_id IN ({ph3})
        """, game_ids)
        matchups = {r[0]: (r[1], r[2], r[3]) for r in cur.fetchall()}

    from src.processing.play_tagger import tag_play

    for pid, desc, gid, clock, player_id, player_name in play_rows:
        t = traits.get(player_id, {})
        if t.get("dog", 0) < args.min_dog: continue
        if t.get("menace", 0) < args.min_menace: continue
        if t.get("unselfish", 0) < args.min_unselfish: continue
        if t.get("tough", 0) < args.min_tough: continue
        if t.get("rim", 0) < args.min_rim: continue
        if t.get("shot", 0) < args.min_shot: continue

        play_tags = tag_play(desc)
        if args.tags and not set(args.tags).issubset(set(play_tags)):
            continue

        home, away, video = matchups.get(gid, ("Unknown", "Unknown", None))
        print(f"{home} vs {away} @ {clock}")
        print(f"  Player: {player_name}")
        print(f"  Traits: {t}")
        print(f"  Tags: {play_tags}")
        print(f"  Play: {desc}")
        print(f"  Video: {video}")
        print("-"*60)

    conn.close()

if __name__ == "__main__":
    main()
