from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List

import chromadb

VECTOR_DB_PATH = os.path.join(os.getcwd(), "data/vector_db")
DB_PATH = os.path.join(os.getcwd(), "data/skout.db")


def _similarity_from_distance(dist: float | None) -> float:
    if dist is None:
        return 0.0
    dist = min(max(float(dist), 0.0), 2.0)
    base = 1.0 - (dist / 2.0)
    return max(0.0, base) ** 0.5


def _lookup_player_meta(name: str, player_id: str | None = None) -> Dict[str, Any]:
    if not name and not player_id:
        return {}
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        if player_id:
            cur.execute(
                "SELECT position, height_in, weight_lb FROM players WHERE player_id = ? LIMIT 1",
                (player_id,),
            )
            row = cur.fetchone()
            if row:
                conn.close()
                return {"position": row[0], "height_in": row[1], "weight_lb": row[2]}
        if name:
            cur.execute(
                "SELECT position, height_in, weight_lb FROM players WHERE full_name = ? LIMIT 1",
                (name,),
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return {"position": row[0], "height_in": row[1], "weight_lb": row[2]}
    except Exception:
        return {}
    return {}


def find_similar_players(player_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not player_name:
        return []
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection(name="skout_plays")

    # find a target embedding by name match
    res = collection.get(where={"player_name": player_name}, include=["embeddings", "metadatas"], limit=1)
    ids = res.get("ids") or []
    if not ids:
        # fallback: find play_id from DB, then fetch embedding by id
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT play_id FROM plays WHERE player_name = ? LIMIT 1", (player_name,))
            row = cur.fetchone()
            if not row:
                last = player_name.split(" ")[-1]
                cur.execute("SELECT play_id FROM plays WHERE player_name LIKE ? LIMIT 1", (f"%{last}%",))
                row = cur.fetchone()
            conn.close()
            if row:
                play_id = row[0]
                res_by_id = collection.get(ids=[play_id], include=["embeddings", "metadatas"])
                emb = res_by_id.get("embeddings")
                if emb is not None and len(emb) > 0:
                    target_emb = emb[0]
                    ids = [play_id]
                else:
                    target_emb = None
            else:
                target_emb = None
        except Exception:
            target_emb = None

        if target_emb is None:
            return []
    else:
        emb = res.get("embeddings")
        if emb is None or len(emb) == 0:
            return []
        target_emb = emb[0]

    # query by embedding
    matches: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for n_results in (max(top_k + 200, 200), 1000):
        qres = collection.query(
            query_embeddings=[target_emb],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        metadatas = qres.get("metadatas", [[]])[0]
        distances = qres.get("distances", [[]])[0]
        for meta, dist in zip(metadatas, distances):
            if not meta:
                continue
            pname = meta.get("player_name") or meta.get("player") or ""
            if not pname or pname == player_name:
                continue
            if pname in seen:
                continue
            seen.add(pname)
            pos = meta.get("position")
            height = meta.get("height_in") or meta.get("height")
            weight = meta.get("weight_lb") or meta.get("weight")
            if not pos or not height or not weight:
                lookup = _lookup_player_meta(pname, meta.get("player_id"))
                pos = pos or lookup.get("position")
                height = height or lookup.get("height_in")
                weight = weight or lookup.get("weight_lb")
            matches.append({
                "player_name": pname,
                "similarity": _similarity_from_distance(dist),
                "position": pos,
                "height_in": height,
                "weight_lb": weight,
            })
            if len(matches) >= top_k:
                break
        if matches:
            break

    return matches
