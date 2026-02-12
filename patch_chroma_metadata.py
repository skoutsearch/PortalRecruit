import os
import sqlite3

import chromadb

VECTOR_DB_PATH = os.path.join(os.getcwd(), "data/vector_db")
DB_PATH = os.path.join(os.getcwd(), "data/skout.db")


def load_player_map(conn):
    cur = conn.cursor()
    try:
        cur.execute("SELECT play_player_id, player_id FROM player_id_map")
        return {row[0]: row[1] for row in cur.fetchall() if row[0] and row[1]}
    except Exception:
        # build mapping if missing
        cur.execute("""
            CREATE TABLE IF NOT EXISTS player_id_map (
                play_player_id TEXT PRIMARY KEY,
                player_id TEXT,
                full_name TEXT
            )
        """)
        cur.execute("""
            INSERT OR REPLACE INTO player_id_map (play_player_id, player_id, full_name)
            SELECT pl.player_id, p.player_id, p.full_name
            FROM plays pl
            JOIN players p ON LOWER(pl.player_name) = LOWER(p.full_name)
            WHERE pl.player_id IS NOT NULL AND p.player_id IS NOT NULL
        """)
        conn.commit()
        cur.execute("SELECT play_player_id, player_id FROM player_id_map")
        return {row[0]: row[1] for row in cur.fetchall() if row[0] and row[1]}


def load_player_meta(conn):
    cur = conn.cursor()
    cur.execute("SELECT player_id, position, height_in, weight_lb FROM players")
    meta = {}
    for pid, pos, h, w in cur.fetchall():
        if not pid:
            continue
        meta[pid] = {
            "position": pos or "",
            "height": int(h) if h is not None else None,
            "weight": int(w) if w is not None else None,
        }
    return meta


def main():
    conn = sqlite3.connect(DB_PATH)
    player_map = load_player_map(conn)
    player_meta = load_player_meta(conn)

    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    col = client.get_collection(name="skout_plays")

    total = 0
    patched = 0
    offset = 0
    limit = 5000
    while True:
        batch = col.get(limit=limit, offset=offset, include=["metadatas"])
        ids = batch.get("ids", [])
        metas = batch.get("metadatas", [])
        if not ids:
            break

        new_metas = []
        for pid, meta in zip(ids, metas):
            total += 1
            meta = meta or {}
            play_player_id = meta.get("player_id")
            canonical = player_map.get(play_player_id)
            m = player_meta.get(canonical or play_player_id)
            if m:
                pos = (m.get("position") or "").upper()
                if pos == "F/C":
                    pos = "C"
                meta["position"] = pos
                if m.get("height") is not None:
                    meta["height"] = int(m.get("height"))
                if m.get("weight") is not None:
                    meta["weight"] = int(m.get("weight"))
                patched += 1
            new_metas.append(meta)

        col.update(ids=ids, metadatas=new_metas)
        offset += limit

    conn.close()
    print(f"Patched {patched} records with Position data.")


if __name__ == "__main__":
    main()
