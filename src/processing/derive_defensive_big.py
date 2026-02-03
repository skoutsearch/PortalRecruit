from __future__ import annotations

import math
import sqlite3
from collections import defaultdict

from src.processing.play_tagger import tag_play
from src.ingestion.db import db_path

RIM_RADIUS = 4.0  # feet (approx)


def _near_rim(x, y) -> bool:
    try:
        if x is None or y is None:
            return False
        # Assume shotX/shotY are feet with rim at (0,0)
        return math.hypot(float(x), float(y)) <= RIM_RADIUS
    except Exception:
        return False


def build_defensive_big_metrics() -> None:
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()

    cur.execute(
        """
        SELECT play_id, description, shotX, shotY, d_player_id, r_player_id
        FROM plays
        WHERE d_player_id IS NOT NULL OR r_player_id IS NOT NULL
        """
    )
    rows = cur.fetchall()

    if not rows:
        conn.close()
        return

    stats = defaultdict(lambda: {
        "def_events": 0,
        "block": 0,
        "rim_contest": 0,
        "def_reb": 0,
    })

    for _, desc, x, y, d_pid, r_pid in rows:
        tags = tag_play(desc or "")
        missed = "missed" in tags
        block = "block" in tags
        rim = _near_rim(x, y)

        if d_pid:
            s = stats[d_pid]
            s["def_events"] += 1
            if block:
                s["block"] += 1
            if rim and missed:
                s["rim_contest"] += 1

        if r_pid:
            stats[r_pid]["def_reb"] += 1

    updates = []
    for pid, s in stats.items():
        total = max(1, s["def_events"])
        block_rate = s["block"] / total
        rim_contest_rate = s["rim_contest"] / total
        defensive_rebound_rate = s["def_reb"] / max(1, s["def_reb"] + (total - s["def_reb"]))

        defensive_big = (
            2.5 * block_rate
            + 2.0 * rim_contest_rate
            + 1.5 * defensive_rebound_rate
        )

        updates.append(
            (
                defensive_big,
                block_rate,
                rim_contest_rate,
                defensive_rebound_rate,
                pid,
            )
        )

    cur.executemany(
        """
        UPDATE player_traits
        SET defensive_big_index = ?,
            block_rate = ?,
            rim_contest_rate = ?,
            defensive_rebound_rate = ?
        WHERE player_id = ?
        """,
        updates,
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    build_defensive_big_metrics()
