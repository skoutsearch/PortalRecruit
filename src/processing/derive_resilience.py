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


def build_resilience_metrics() -> None:
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    quarter_col = _choose_quarter_column(cur)

    cur.execute(
        f"""
        SELECT player_id, description, {quarter_col}, clock_seconds,
               short_clock, eob, press, zone, hard_double,
               home_score, away_score, is_home
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
        "made": 0,
        "turnover": 0,
        "trailing": 0,
        "trailing_made": 0,
        "short": 0,
        "short_made": 0,
        "eob": 0,
        "eob_made": 0,
        "press": 0,
        "press_success": 0,
        "zone": 0,
        "zone_success": 0,
        "hard": 0,
        "hard_success": 0,
        "clutch": 0,
        "clutch_made": 0,
    })

    for (player_id, desc, quarter, clock_sec, short_clock, eob, press, zone,
         hard_double, home_score, away_score, is_home) in rows:
        tags = tag_play(desc or "")
        made_or_score = "made" in tags or "score" in tags
        turnover = "turnover" in tags

        s = stats[player_id]
        s["total"] += 1
        if made_or_score:
            s["made"] += 1
        if turnover:
            s["turnover"] += 1

        # trailing context
        if home_score is not None and away_score is not None and is_home is not None:
            team_score = home_score if is_home else away_score
            opp_score = away_score if is_home else home_score
            if team_score < opp_score:
                s["trailing"] += 1
                if made_or_score:
                    s["trailing_made"] += 1

        if short_clock:
            s["short"] += 1
            if made_or_score:
                s["short_made"] += 1
        if eob:
            s["eob"] += 1
            if made_or_score:
                s["eob_made"] += 1
        if press:
            s["press"] += 1
            if made_or_score and not turnover:
                s["press_success"] += 1
        if zone:
            s["zone"] += 1
            if made_or_score:
                s["zone_success"] += 1
        if hard_double:
            s["hard"] += 1
            if made_or_score:
                s["hard_success"] += 1

        # clutch: 4Q last 2 minutes
        if quarter == 4 and isinstance(clock_sec, int) and clock_sec <= 120:
            s["clutch"] += 1
            if made_or_score:
                s["clutch_made"] += 1

    updates = []
    for pid, s in stats.items():
        total = max(1, s["total"])
        trailing_make_rate = s["trailing_made"] / max(1, s["trailing"])
        short_clock_make_rate = s["short_made"] / max(1, s["short"])
        eob_make_rate = s["eob_made"] / max(1, s["eob"])
        press_success_rate = s["press_success"] / max(1, s["press"])
        zone_success_rate = s["zone_success"] / max(1, s["zone"])
        hard_double_success_rate = s["hard_success"] / max(1, s["hard"])
        clutch_make_rate = s["clutch_made"] / max(1, s["clutch"])
        turnover_rate = s["turnover"] / total

        resilience = (
            2.0 * trailing_make_rate
            + 1.8 * short_clock_make_rate
            + 1.5 * eob_make_rate
            + 1.2 * press_success_rate
            + 1.2 * zone_success_rate
            + 1.0 * hard_double_success_rate
            + 1.5 * clutch_make_rate
            - 1.5 * turnover_rate
        )

        updates.append(
            (
                resilience,
                trailing_make_rate,
                short_clock_make_rate,
                eob_make_rate,
                press_success_rate,
                zone_success_rate,
                hard_double_success_rate,
                clutch_make_rate,
                pid,
            )
        )

    cur.executemany(
        """
        UPDATE player_traits
        SET resilience_index = ?,
            trailing_make_rate = ?,
            short_clock_make_rate = ?,
            eob_make_rate = ?,
            press_success_rate = ?,
            zone_success_rate = ?,
            hard_double_success_rate = ?,
            clutch_make_rate = ?
        WHERE player_id = ?
        """,
        updates,
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    build_resilience_metrics()
