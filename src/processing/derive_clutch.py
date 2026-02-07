from __future__ import annotations

import sqlite3
from collections import defaultdict

from src.processing.play_tagger import tag_play
from src.ingestion.db import db_path


def _choose_quarter_column(cur: sqlite3.Cursor) -> str:
    cur.execute("PRAGMA table_info(plays)")
    cols = {r[1] for r in cur.fetchall()}
    if "gameQuarter" in cols:
        return "gameQuarter"
    if "period" in cols:
        return "period"
    raise RuntimeError("plays table missing both gameQuarter and period columns")


def build_clutch_metrics() -> None:
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    quarter_col = _choose_quarter_column(cur)

    cur.execute(
        f"""
        SELECT player_id, description, {quarter_col}, clock_seconds,
               short_clock, ato, home_score, away_score, is_home,
               assist_player_id
        FROM plays
        WHERE player_id IS NOT NULL
        """
    )
    rows = cur.fetchall()

    if not rows:
        conn.close()
        return

    stats = defaultdict(lambda: {
        "clutch_total": 0,
        "clutch_make": 0,
        "clutch_assist": 0,
        "clutch_deflection": 0,
        "clutch_turnover": 0,
        "ato_clutch_make": 0,
        "short_clock_clutch_make": 0,
    })

    for (player_id, desc, quarter, clock_sec, short_clock, ato,
         home_score, away_score, is_home, assist_pid) in rows:
        tags = tag_play(desc or "")
        made_or_score = "made" in tags or "score" in tags
        turnover = "turnover" in tags
        deflection = "deflection" in tags

        close_game = False
        if home_score is not None and away_score is not None:
            if abs(home_score - away_score) <= 5:
                close_game = True

        is_clutch = (quarter == 4 and isinstance(clock_sec, int) and clock_sec <= 240 and close_game)
        if not is_clutch:
            continue

        s = stats[player_id]
        s["clutch_total"] += 1
        if made_or_score:
            s["clutch_make"] += 1
        if turnover:
            s["clutch_turnover"] += 1
        if deflection:
            s["clutch_deflection"] += 1

        if assist_pid:
            stats[assist_pid]["clutch_assist"] += 1

        if ato and made_or_score:
            s["ato_clutch_make"] += 1
        if short_clock and made_or_score:
            s["short_clock_clutch_make"] += 1

    updates = []
    for pid, s in stats.items():
        total = max(1, s["clutch_total"])
        clutch_make_rate = s["clutch_make"] / total
        clutch_assist_rate = s["clutch_assist"] / total
        clutch_deflection_rate = s["clutch_deflection"] / total
        clutch_turnover_rate = s["clutch_turnover"] / total

        clutch_index = (
            2.5 * clutch_make_rate
            + 1.8 * clutch_assist_rate
            + 1.0 * clutch_deflection_rate
            - 2.0 * clutch_turnover_rate
        )

        updates.append(
            (
                clutch_index,
                clutch_make_rate,
                clutch_assist_rate,
                clutch_deflection_rate,
                clutch_turnover_rate,
                pid,
            )
        )

    cur.executemany(
        """
        UPDATE player_traits
        SET clutch_index = ?,
            clutch_make_rate = ?,
            clutch_assist_rate = ?,
            clutch_deflection_rate = ?,
            clutch_turnover_rate = ?
        WHERE player_id = ?
        """,
        updates,
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    build_clutch_metrics()
