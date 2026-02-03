#!/usr/bin/env python3
"""Incremental ingest using only API-accessible data.

Single-path, best-practice flow:
1) Discover API access (seasons -> teams -> games)
2) Compare against local DB (if exists)
3) Ingest ONLY missing games / events / players
4) Backfill names, rebuild traits, regenerate embeddings

Rate limiting is handled by SynergyClient._get (retry + backoff on 429).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.db import connect_db, ensure_schema
from src.ingestion.pipeline import (
    _unwrap_list_payload,
    iter_games,
    upsert_games,
    upsert_players,
    upsert_plays,
)
from src.ingestion.synergy_client import SynergyClient


def _season_ids(seasons: list[dict]) -> list[str]:
    out: list[str] = []
    for s in seasons:
        for key in ("id", "seasonId", "seasonID"):
            if s.get(key):
                out.append(str(s.get(key)))
                break
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    print("🔎 Checking API access...")
    client = SynergyClient()

    seasons_payload = client.get_seasons("ncaamb")
    seasons = [s for s in _unwrap_list_payload(seasons_payload) if isinstance(s, dict)]
    if not seasons:
        print(f"❌ No seasons available (status={client.last_status_code}, err={client.last_error})")
        return 1

    season_ids = _season_ids(seasons)
    if not season_ids:
        print("❌ Unable to determine any season_id from API.")
        return 1

    conn = connect_db()
    ensure_schema(conn)
    cur = conn.cursor()

    total_new_games = 0
    total_new_plays = 0

    acc_2021_22 = {
        "BostonCollege",
        "Clemson",
        "Duke",
        "FloridaState",
        "GeorgiaTech",
        "Louisville",
        "MiamiFL",
        "NorthCarolina",
        "NorthCarolinaState",
        "NotreDame",
        "Pittsburgh",
        "Syracuse",
        "Virginia",
        "VirginiaTech",
        "WakeForest",
    }

    for season_id in season_ids:
        teams_payload = client.get_teams("ncaamb", season_id)
        teams = [t for t in _unwrap_list_payload(teams_payload) if isinstance(t, dict)]
        team_ids = [t.get("id") for t in teams if t.get("id")]

        if not team_ids:
            print(f"⚠️  No teams accessible for season {season_id}")
            continue

        # Existing games for this season
        cur.execute("SELECT game_id FROM games WHERE season_id = ?", (season_id,))
        existing_game_ids = {r[0] for r in cur.fetchall()}

        print(f"✅ Season {season_id} | teams: {len(team_ids)} | existing games: {len(existing_game_ids)}")

        # Fetch all accessible games (by team), keep only missing ones
        new_games: dict[str, dict] = {}
        for tid in team_ids:
            for g in iter_games(client, "ncaamb", season_id, tid):
                gid = g.get("id")
                if not gid or gid in existing_game_ids or gid in new_games:
                    continue
                new_games[gid] = g

        if new_games:
            inserted = upsert_games(conn, season_id, list(new_games.values()))
            total_new_games += inserted
        else:
            print(f"ℹ️  No new games for season {season_id}")

        # Players: fetch only if team missing in DB
        for tid in team_ids:
            cur.execute("SELECT 1 FROM players WHERE team_id = ? LIMIT 1", (tid,))
            if cur.fetchone():
                continue
            payload = client.get_team_players("ncaamb", tid)
            players = [p for p in _unwrap_list_payload(payload) if isinstance(p, dict)]
            if players:
                upsert_players(conn, tid, players)

        # Events: focus on ACC games with zero or low plays
        cur.execute(
            """
            SELECT g.game_id, g.home_team, g.away_team, COUNT(p.play_id) AS plays
            FROM games g
            LEFT JOIN plays p ON p.game_id = g.game_id
            WHERE g.season_id = ?
              AND (g.home_team IN ({}) OR g.away_team IN ({}))
            GROUP BY g.game_id
            HAVING plays = 0 OR plays < 50
            """.format(
                ",".join(["?"] * len(acc_2021_22)), ",".join(["?"] * len(acc_2021_22))
            ),
            (season_id, *acc_2021_22, *acc_2021_22),
        )
        game_ids_to_fill = [r[0] for r in cur.fetchall()]

        if game_ids_to_fill:
            for idx, gid in enumerate(game_ids_to_fill):
                if idx % 10 == 0:
                    print(f"   events {idx}/{len(game_ids_to_fill)}")
                payload = client.get_game_events("ncaamb", gid)
                if not payload:
                    continue
                events = [e for e in _unwrap_list_payload(payload) if isinstance(e, dict)]
                total_new_plays += upsert_plays(conn, gid, events)

    conn.close()

    print(f"✅ New games: {total_new_games} | New plays: {total_new_plays}")

    # Backfill player names
    try:
        from scripts.backfill_player_names import main as backfill_names
        backfill_names()
    except Exception as e:
        print(f"⚠️ Backfill failed: {e}")

    # Rebuild traits (post-backfill)
    try:
        from src.processing.derive_player_traits import build_player_traits
        build_player_traits()
    except Exception as e:
        print(f"⚠️ Trait build failed: {e}")

    # Always regenerate embeddings to reflect any new plays
    try:
        from src.processing.generate_embeddings import generate_embeddings
        generate_embeddings()
    except Exception as e:
        print(f"⚠️ Embeddings failed: {e}")

    print("✅ Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
