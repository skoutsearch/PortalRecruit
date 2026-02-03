import os
import sqlite3

from src.processing.play_tagger import tag_play

DB_PATH = os.path.join(os.getcwd(), "data/skout.db")

# Simple keyword-based proxies until we have richer event typing
DOG_KEYWORDS = [
    "offensive rebound", "off. rebound", "oreb",
    "steal", "block", "charge", "loose ball", "dive"
]


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
            shot_making_index REAL
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

    # Aggregate counts in Python (simple + fast enough for now)
    agg = {}
    for player_id, player_name, desc in rows:
        if player_id not in agg:
            agg[player_id] = {
                "player_name": player_name,
                "dog_events": 0,
                "menace_events": 0,
                "tough_events": 0,
                "rim_events": 0,
                "assists": 0,
                "turnovers": 0,
                "made": 0,
                "missed": 0,
                "total_events": 0,
            }
        tags = set(tag_play(desc))
        agg[player_id]["total_events"] += 1
        agg[player_id]["dog_events"] += int(bool(tags & {"oreb", "loose_ball", "charge_taken", "deflection"}))
        agg[player_id]["menace_events"] += int(bool(tags & {"steal", "block", "deflection"}))
        agg[player_id]["tough_events"] += int(bool(tags & {"oreb", "loose_ball", "charge_taken"}))
        agg[player_id]["rim_events"] += int(bool(tags & {"drive", "rim_pressure"}))
        agg[player_id]["assists"] += int("assist" in tags)
        agg[player_id]["turnovers"] += int("turnover" in tags)
        agg[player_id]["made"] += int("made" in tags)
        agg[player_id]["missed"] += int("missed" in tags)

    # Compute indices and persist
    for pid, data in agg.items():
        total = max(1, data["total_events"])
        dog_index = round((data["dog_events"] / total) * 100, 3)
        menace_index = round((data["menace_events"] / total) * 100, 3)
        toughness_index = round((data["tough_events"] / total) * 100, 3)
        rim_pressure_index = round((data["rim_events"] / total) * 100, 3)
        unselfish_index = round((data["assists"] / max(1, data["assists"] + data["turnovers"])) * 100, 3)
        shot_making_index = round((data["made"] / max(1, data["made"] + data["missed"])) * 100, 3)

        cur.execute(
            """
            INSERT OR REPLACE INTO player_traits
            (player_id, player_name, dog_events, total_events, dog_index,
             menace_index, unselfish_index, toughness_index, rim_pressure_index, shot_making_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )

    conn.commit()
    conn.close()
    print(f"✅ player_traits updated for {len(agg)} players")


if __name__ == "__main__":
    build_player_traits()
