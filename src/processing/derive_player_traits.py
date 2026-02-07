import json
import os
import sqlite3
import sys
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.processing.play_tagger import tag_play

DB_PATH = os.path.join(os.getcwd(), "data/skout.db")

# Simple keyword-based proxies until we have richer event typing
DOG_KEYWORDS = [
    "offensive rebound", "off. rebound", "oreb",
    "steal", "block", "charge", "loose ball", "dive"
]

GRAVITY_KEYWORDS = [
    "double team", "double-team", "double", "trap", "face guard", "face-guard",
    "deny", "denied", "decoy", "gravity", "magnet", "overhelp", "over-help",
    "helped off", "stay glued", "stayed glued", "tag", "hard hedge", "hedge",
    "flare screen", "flare", "pin down", "pin-down", "stagger", "ghost screen",
    "ghost", "slip screen", "slip", "handoff", "dho",
]

DOG_TAGS = ["oreb", "loose_ball", "charge_taken", "deflection", "steal", "block"]
DEFAULT_DOG_WEIGHTS = {
    "oreb": 1.5,
    "steal": 2.0,
    "block": 1.5,
    "charge_taken": 3.0,
    "loose_ball": 1.0,
    "deflection": 1.0,
}
DOG_WEIGHTS_PATH = REPO_ROOT / "models" / "dog_index_weights.json"


def _load_dog_weights() -> dict:
    if DOG_WEIGHTS_PATH.exists():
        try:
            with open(DOG_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            weights = data.get("weights", data)
            return {k: float(weights.get(k, DEFAULT_DOG_WEIGHTS.get(k, 1.0))) for k in DOG_TAGS}
        except Exception:
            return DEFAULT_DOG_WEIGHTS
    return DEFAULT_DOG_WEIGHTS


def _count_keywords(desc: str, keywords: list[str]) -> int:
    if not desc:
        return 0
    d = desc.lower()
    return sum(1 for k in keywords if k in d)


def build_player_traits():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create table to store traits (idempotent)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_traits (
            player_id TEXT PRIMARY KEY,
            player_name TEXT,
            dog_events INTEGER,
            total_events INTEGER,
            dog_index REAL,
            menace_index REAL,
            unselfish_index REAL,
            toughness_index REAL,
            rim_pressure_index REAL,
            shot_making_index REAL,
            gravity_index REAL
        )
        """
    )

    # Ensure new columns exist (safe ALTER)
    for col, ctype in [
        ("menace_index", "REAL"),
        ("unselfish_index", "REAL"),
        ("toughness_index", "REAL"),
        ("rim_pressure_index", "REAL"),
        ("shot_making_index", "REAL"),
        ("size_index", "REAL"),
        ("gravity_index", "REAL"),
    ]:
        try:
            cur.execute(f"ALTER TABLE player_traits ADD COLUMN {col} {ctype}")
        except Exception:
            pass

    # Pull plays with player_id
    cur.execute(
        """
        SELECT player_id, player_name, description
        FROM plays
        WHERE player_id IS NOT NULL
        """
    )
    rows = cur.fetchall()

    # Pull size data if available
    size_map = {}
    try:
        cur.execute("SELECT player_id, height_in, weight_lb FROM players")
        size_rows = cur.fetchall()
        size_map = {r[0]: (r[1], r[2]) for r in size_rows}
    except Exception:
        size_map = {}

    # Aggregate counts in Python (simple + fast enough for now)
    agg = {}
    for player_id, player_name, desc in rows:
        if player_id not in agg:
            agg[player_id] = {
                "player_name": player_name,
                "dog_events": 0,
                "dog_tag_counts": {k: 0 for k in DOG_TAGS},
                "menace_events": 0,
                "tough_events": 0,
                "rim_events": 0,
                "assists": 0,
                "turnovers": 0,
                "made": 0,
                "missed": 0,
                "total_events": 0,
                "gravity_events": 0,
            }
        tags = set(tag_play(desc))
        if "non_possession" in tags:
            continue

        agg[player_id]["total_events"] += 1
        dog_hit = False
        for tag in DOG_TAGS:
            if tag in tags:
                agg[player_id]["dog_tag_counts"][tag] += 1
                dog_hit = True
        agg[player_id]["dog_events"] += int(dog_hit)
        agg[player_id]["menace_events"] += int(bool(tags & {"steal", "block", "deflection"}))
        agg[player_id]["tough_events"] += int(bool(tags & {"oreb", "loose_ball", "charge_taken"}))
        agg[player_id]["rim_events"] += int(bool(tags & {"drive", "rim_pressure"}))
        agg[player_id]["assists"] += int("assist" in tags)
        agg[player_id]["turnovers"] += int("turnover" in tags)
        agg[player_id]["made"] += int("made" in tags)
        agg[player_id]["missed"] += int("missed" in tags)

        gravity_keyword_hit = _count_keywords(desc, GRAVITY_KEYWORDS) > 0
        gravity_pnr_pull = "pnr" in tags and ("pull_up" in tags or "3pt" in tags)
        gravity_spacing = "assist" in tags and "3pt" in tags
        gravity_handoff = "handoff" in tags and "3pt" in tags
        gravity_signal = gravity_keyword_hit or gravity_pnr_pull or gravity_spacing or gravity_handoff
        agg[player_id]["gravity_events"] += int(gravity_signal)

    # Compute indices and persist
    dog_weights = _load_dog_weights()
    for pid, data in agg.items():
        total = max(1, data["total_events"])
        dog_score = sum(data["dog_tag_counts"].get(tag, 0) * dog_weights.get(tag, 1.0) for tag in DOG_TAGS)
        dog_index = round((dog_score / total) * 100, 3)
        menace_index = round((data["menace_events"] / total) * 100, 3)
        toughness_index = round((data["tough_events"] / total) * 100, 3)
        rim_pressure_index = round((data["rim_events"] / total) * 100, 3)
        unselfish_index = round((data["assists"] / max(1, data["assists"] + data["turnovers"])) * 100, 3)
        shot_making_index = round((data["made"] / max(1, data["made"] + data["missed"])) * 100, 3)
        gravity_index = round((data["gravity_events"] / total) * 100, 3)

        # size index: normalize height/weight to a 0-100-ish scale if present
        h, w = size_map.get(pid, (None, None))
        size_index = None
        if h is not None or w is not None:
            h_val = float(h) if h is not None else 0.0
            w_val = float(w) if w is not None else 0.0
            size_index = round(min(100.0, (h_val * 1.0) + (w_val * 0.1)), 3)

        cur.execute(
            """
            INSERT OR REPLACE INTO player_traits
            (player_id, player_name, dog_events, total_events, dog_index,
             menace_index, unselfish_index, toughness_index, rim_pressure_index,
             shot_making_index, size_index, gravity_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                data["player_name"],
                data["dog_events"],
                data["total_events"],
                dog_index,
                menace_index,
                unselfish_index,
                toughness_index,
                rim_pressure_index,
                shot_making_index,
                size_index,
                gravity_index,
            ),
        )

    conn.commit()
    conn.close()
    print(f"✅ player_traits updated for {len(agg)} players")


if __name__ == "__main__":
    build_player_traits()
