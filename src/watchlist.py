from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import chromadb

from src.search.semantic import semantic_search

WATCHLIST_PATH = os.path.join(os.getcwd(), "data/saved_searches.json")
VECTOR_DB_PATH = os.path.join(os.getcwd(), "data/vector_db")


def _load_watchlist() -> list[dict]:
    if not os.path.exists(WATCHLIST_PATH):
        return []
    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_watchlist(data: list[dict]) -> None:
    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_search(name: str, query_params: dict) -> bool:
    if not name:
        return False
    watchlist = _load_watchlist()
    updated = False
    for item in watchlist:
        if item.get("name") == name:
            item["params"] = query_params
            updated = True
            break
    if not updated:
        watchlist.append({"name": name, "params": query_params})
    _save_watchlist(watchlist)
    return True


def get_saved_searches() -> list[dict]:
    return _load_watchlist()


def delete_search(name: str) -> bool:
    watchlist = _load_watchlist()
    filtered = [w for w in watchlist if w.get("name") != name]
    if len(filtered) == len(watchlist):
        return False
    _save_watchlist(filtered)
    return True


def _load_shortlist_names() -> set[str]:
    try:
        with open(os.path.join(os.getcwd(), "data/shortlist.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return set()
        return {str(p.get("name") or "").strip() for p in data if p.get("name")}
    except Exception:
        return set()


def check_for_alerts(search_name: str) -> int:
    watchlist = _load_watchlist()
    found = next((w for w in watchlist if w.get("name") == search_name), None)
    if not found:
        return 0
    params = found.get("params") or {}
    query = params.get("query") or ""
    if not query:
        return 0
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection(name="skout_plays")
    play_ids, breakdowns = semantic_search(collection, query=query, n_results=5, return_breakdowns=True)
    names = []
    if breakdowns:
        for pid in play_ids:
            meta = breakdowns.get(pid) or {}
            name = meta.get("player_name") or meta.get("name")
            if name:
                names.append(name)
    if not names and play_ids:
        res = collection.get(ids=play_ids, include=["metadatas"])
        for meta in res.get("metadatas") or []:
            name = meta.get("player_name") or meta.get("name")
            if name:
                names.append(name)
    shortlist = _load_shortlist_names()
    new_matches = [n for n in names if n not in shortlist]
    return len(set(new_matches))
