from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import numpy as np
import sqlite3
from dotenv import load_dotenv

from src.search.semantic import semantic_search

root_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(root_dir / ".env")

VECTOR_DB_PATH = os.path.join(os.getcwd(), "data/vector_db")
DB_PATH = os.path.join(os.getcwd(), "data/skout.db")
CLUSTER_MAP_PATH = os.path.join(os.getcwd(), "data/cluster_map.json")
CLUSTER_LABELS_PATH = os.path.join(os.getcwd(), "data/cluster_labels.json")


def _load_collection():
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    return client.get_collection(name="skout_plays")


def _get_player_name(pid: str) -> str | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT full_name FROM players WHERE player_id = ? LIMIT 1", (pid,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def discover_archetypes(n_clusters: int = 8) -> dict:
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return {}

    collection = _load_collection()
    res = collection.get(include=["embeddings", "metadatas"])
    embeddings = res.get("embeddings")
    if embeddings is None:
        embeddings = []
    metas = res.get("metadatas")
    if metas is None:
        metas = []

    player_embeds: dict[str, list] = {}
    for emb, meta in zip(embeddings, metas):
        if meta is None:
            continue
        pid = meta.get("player_id") or meta.get("player") or meta.get("playerID")
        if not pid:
            continue
        player_embeds.setdefault(pid, []).append(emb)

    player_ids = []
    vectors = []
    for pid, embs in player_embeds.items():
        if not embs:
            continue
        player_ids.append(pid)
        vectors.append(np.mean(np.array(embs), axis=0))

    if not vectors:
        return {}

    kmeans = KMeans(n_clusters=min(n_clusters, len(vectors)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(np.array(vectors))

    cluster_map = {pid: int(label) for pid, label in zip(player_ids, labels)}
    with open(CLUSTER_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(cluster_map, f, indent=2)
    return cluster_map


def _load_cluster_map() -> dict:
    if not os.path.exists(CLUSTER_MAP_PATH):
        return {}
    try:
        return json.loads(open(CLUSTER_MAP_PATH, "r", encoding="utf-8").read())
    except Exception:
        return {}


def name_clusters(cluster_map: dict) -> dict:
    if not cluster_map:
        return {}
    clusters: dict[int, list[str]] = {}
    for pid, cid in cluster_map.items():
        clusters.setdefault(int(cid), []).append(pid)

    labels: dict[int, str] = {}
    api_key = os.getenv("OPENAI_API_KEY")
    fallback_names = [
        "Floor General",
        "Paint Beast",
        "Sniper",
        "Switchable Wing",
        "Rim Runner",
        "Heliocentric Guard",
        "Connector Big",
        "Microwave Scorer",
    ]

    for cid, pids in clusters.items():
        sample = pids[:5]
        players = []
        for pid in sample:
            name = _get_player_name(pid) or pid
            players.append(name)

        if api_key:
            try:
                import requests
                prompt = (
                    "Here are 5 players in a cluster. Give this playing style a creative 2-word name like "
                    "'Rim Runner' or 'Heliocentric Guard'.\n\nPlayers: " + ", ".join(players)
                )
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.5,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                label = resp.json()["choices"][0]["message"]["content"].strip()
                labels[cid] = label
                continue
            except Exception:
                pass

        labels[cid] = fallback_names[int(cid) % len(fallback_names)]

    with open(CLUSTER_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    return labels


def get_cluster_label(player_id: str) -> str | None:
    cluster_map = _load_cluster_map()
    if not cluster_map:
        return None
    labels = {}
    if os.path.exists(CLUSTER_LABELS_PATH):
        try:
            labels = json.loads(open(CLUSTER_LABELS_PATH, "r", encoding="utf-8").read())
        except Exception:
            labels = {}
    cid = cluster_map.get(player_id)
    if cid is None:
        return None
    return labels.get(str(cid)) or labels.get(int(cid)) or f"Cluster {cid}"
