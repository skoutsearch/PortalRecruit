from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.db import connect_db, ensure_schema
from src.processing.play_tagger import tag_play


def _parse_lineup(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def _lineup_player_ids(raw) -> list[str]:
    lineup = _parse_lineup(raw)
    ids = []
    for item in lineup:
        if isinstance(item, dict):
            pid = item.get("id") or item.get("playerId") or item.get("player_id")
            if pid:
                ids.append(str(pid))
        elif isinstance(item, str):
            ids.append(item)
    return ids


def _points_from_tags(tags: list[str], desc: str) -> int:
    if "ft" in tags:
        return 1 if "made" in tags else 0
    if "3pt" in tags:
        return 3 if "made" in tags else 0
    if "made" in tags:
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill minutes + boxscore stats from play-by-play.")
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "skout.db"))
    parser.add_argument("--season", default=None, help="Optional season_id filter")
    args = parser.parse_args()

    conn = connect_db()
    ensure_schema(conn)
    cur = conn.cursor()

    query = """
    SELECT p.game_id, g.season_id, p.description, p.clock_seconds,
           p.player_id, p.assist_player_id, p.r_player_id, p.d_player_id,
           p.duration, p.offensive_lineup
    FROM plays p
    JOIN games g ON g.game_id = p.game_id
    """
    params = []
    if args.season:
        query += " WHERE g.season_id = ?"
        params.append(args.season)

    rows = cur.execute(query, params).fetchall()

    stats = defaultdict(lambda: {
        "minutes": 0.0,
        "reb": 0,
        "ast": 0,
        "stl": 0,
        "blk": 0,
    })

    for (
        _game_id,
        season_id,
        desc,
        clock_seconds,
        player_id,
        assist_id,
        rebound_id,
        defender_id,
        duration,
        offensive_lineup,
    ) in rows:
        if not season_id:
            continue

        tags = tag_play(desc or "", clock_seconds)

        # Minutes (best-effort): use offensive lineup duration if available
        if duration is not None:
            lineup_ids = _lineup_player_ids(offensive_lineup)
            if lineup_ids:
                for pid in lineup_ids:
                    key = (pid, season_id)
                    stats[key]["minutes"] += float(duration) / 60.0
            else:
                # Fallback: attribute duration to involved players only
                for pid in {player_id, assist_id, rebound_id, defender_id}:
                    if pid:
                        key = (pid, season_id)
                        stats[key]["minutes"] += float(duration) / 60.0

        # Rebounds
        if "rebound" in tags:
            pid = rebound_id or player_id
            if pid:
                stats[(pid, season_id)]["reb"] += 1

        # Assists
        if "assist" in tags and assist_id:
            stats[(assist_id, season_id)]["ast"] += 1

        # Steals
        if "steal" in tags:
            pid = defender_id or player_id
            if pid:
                stats[(pid, season_id)]["stl"] += 1

        # Blocks
        if "block" in tags:
            pid = defender_id or player_id
            if pid:
                stats[(pid, season_id)]["blk"] += 1

    if not stats:
        print("No stats to backfill.")
        return

    now = datetime.utcnow().isoformat()

    # Upsert into player_season_stats (only the new columns)
    for (pid, season_id), vals in stats.items():
        cur.execute(
            """
            INSERT INTO player_season_stats (player_id, season_id, minutes, reb, ast, stl, blk, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id, season_id)
            DO UPDATE SET
                minutes = COALESCE(player_season_stats.minutes, 0) + excluded.minutes,
                reb = COALESCE(player_season_stats.reb, 0) + excluded.reb,
                ast = COALESCE(player_season_stats.ast, 0) + excluded.ast,
                stl = COALESCE(player_season_stats.stl, 0) + excluded.stl,
                blk = COALESCE(player_season_stats.blk, 0) + excluded.blk,
                updated_at = excluded.updated_at
            """,
            (
                pid,
                season_id,
                round(vals["minutes"], 3),
                vals["reb"],
                vals["ast"],
                vals["stl"],
                vals["blk"],
                now,
            ),
        )

    conn.commit()
    conn.close()

    print(f"✅ Backfilled {len(stats)} player-season rows from plays")


if __name__ == "__main__":
    main()
