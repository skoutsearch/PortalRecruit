import os
import sqlite3

import chromadb

from src.search.semantic import semantic_search

VECTOR_DB_PATH = os.path.join(os.getcwd(), "data/vector_db")
DB_PATH = os.path.join(os.getcwd(), "data/skout.db")


def search_plays(query, n_results=5):
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection(name="skout_plays")
    play_ids = semantic_search(collection, query=query, n_results=n_results)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"\n🔍 Search Results for: '{query}'")
    print("-" * 50)

    if not play_ids:
        print("No results found.")
        return

    for i, play_id in enumerate(play_ids):
        cursor.execute(
            """
            SELECT description, tags, clock_display, game_id
            FROM plays
            WHERE play_id = ?
            """,
            (play_id,),
        )
        row = cursor.fetchone()
        if not row:
            continue
        desc, tags, clock, game_id = row

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

        print(f"[{i+1}] {matchup} @ {clock}")
        print(f" Play: {desc}")
        print(f" Tags: [{tags}]")
        print(f" File: {v_path}")
        print("")


if __name__ == "__main__":
    while True:
        q = input("Enter search query (or 'q' to quit): ")
        if q.lower() == "q":
            break
        search_plays(q)
