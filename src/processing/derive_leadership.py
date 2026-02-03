from __future__ import annotations

import sqlite3
from collections import defaultdict

from src.processing.play_tagger import tag_play
from src.ingestion.db import db_path


def build_leadership_metrics() -> None:
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()

    cur.execute(
        """
        SELECT play_id, player_id, description, ato, short_clock, eob, heave,
               press, zone, hard_double, assist_player_id
        FROM plays
        WHERE player_id IS NOT NULL
        """
    )
    rows = cur.fetchall()

    if not rows:
        conn.close()
        return

    stats = defaultdict(lambda: {
        "total": 0,
        "assist": 0,
        "turnover": 0,
        "ato": 0,
        "ato_success": 0,
        "short_clock": 0,
        "short_clock_success": 0,
        "eob": 0,
        "eob_success": 0,
        "press": 0,
        "press_success": 0,
        "zone": 0,
        "zone_success": 0,
        "hard_double": 0,
        "hard_double_success": 0,
        "heave": 0,
    })

    for _, player_id, desc, ato, short_clock, eob, heave, press, zone, hard_double, assist_pid in rows:
        tags = tag_play(desc or "")
        made_or_score = "made" in tags or "score" in tags
        turnover = "turnover" in tags

        s = stats[player_id]
        s["total"] += 1
        if turnover:
            s["turnover"] += 1

        # assist credit
        if assist_pid:
            stats[assist_pid]["assist"] += 1

        if ato:
            s["ato"] += 1
            if made_or_score:
                s["ato_success"] += 1
        if short_clock:
            s["short_clock"] += 1
            if made_or_score:
                s["short_clock_success"] += 1
        if eob:
            s["eob"] += 1
            if made_or_score:
                s["eob_success"] += 1
        if press:
            s["press"] += 1
            if made_or_score and not turnover:
                s["press_success"] += 1
        if zone:
            s["zone"] += 1
            if made_or_score:
                s["zone_success"] += 1
        if hard_double:
            s["hard_double"] += 1
            if made_or_score:
                s["hard_double_success"] += 1
        if heave:
            s["heave"] += 1

    # Update player_traits with rates + leadership_index
    updates = []
    for pid, s in stats.items():
        total = max(1, s["total"])
        assist_rate = s["assist"] / total
        turnover_rate = s["turnover"] / total
        ato_rate = s["ato"] / total
        short_clock_rate = s["short_clock"] / total
        eob_rate = s["eob"] / total
        press_rate = s["press"] / total
        zone_rate = s["zone"] / total
        hard_double_rate = s["hard_double"] / total
        heave_rate = s["heave"] / total

        ato_success = (s["ato_success"] / max(1, s["ato"]))
        short_success = (s["short_clock_success"] / max(1, s["short_clock"]))
        eob_success = (s["eob_success"] / max(1, s["eob"]))
        press_success = (s["press_success"] / max(1, s["press"]))
        zone_success = (s["zone_success"] / max(1, s["zone"]))
        hard_success = (s["hard_double_success"] / max(1, s["hard_double"]))

        # Leadership index (data-only proxy)
        leadership = (
            2.5 * assist_rate
            + 2.0 * ato_success
            + 2.0 * short_success
            + 1.5 * eob_success
            + 1.2 * press_success
            + 1.2 * zone_success
            + 1.0 * hard_success
            - 2.0 * turnover_rate
            - 0.5 * heave_rate
        )

        updates.append(
            (
                leadership,
                ato_rate,
                short_clock_rate,
                eob_rate,
                press_rate,
                zone_rate,
                hard_double_rate,
                assist_rate,
                turnover_rate,
                pid,
            )
        )

    cur.executemany(
        """
        UPDATE player_traits
        SET leadership_index = ?,
            ato_rate = ?,
            short_clock_rate = ?,
            eob_rate = ?,
            press_rate = ?,
            zone_rate = ?,
            hard_double_rate = ?,
            assist_rate = ?,
            turnover_rate = ?
        WHERE player_id = ?
        """,
        updates,
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    build_leadership_metrics()
