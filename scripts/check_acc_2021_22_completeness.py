#!/usr/bin/env python3
"""Verify ACC 2021-22 completeness vs API-accessible data.

Compares:
- API-accessible game_ids for ACC teams
- DB game_ids for those teams
- Event counts per game (API vs DB plays)

NOTE: This script makes many API calls and will respect SynergyClient rate limiting.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict

from src.ingestion.pipeline import _unwrap_list_payload, iter_games
from src.ingestion.synergy_client import SynergyClient

SEASON_ID = "6085b5d0e6c2413bc4ba9122"
ACC_TEAMS = {
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


def norm(name: str) -> str:
    return "".join(c for c in name if c.isalnum()).lower()


def main() -> int:
    client = SynergyClient()

    # Resolve ACC team IDs from API
    teams_payload = client.get_teams("ncaamb", SEASON_ID)
    teams = [t for t in _unwrap_list_payload(teams_payload) if isinstance(t, dict)]
    acc_norm = {norm(t) for t in ACC_TEAMS}

    acc_team_ids = []
    for t in teams:
        name = t.get("name") or t.get("market") or t.get("shortName") or ""
        if norm(name) in acc_norm:
            acc_team_ids.append(t.get("id"))

    acc_team_ids = [tid for tid in acc_team_ids if tid]

    if len(acc_team_ids) < len(ACC_TEAMS):
        print("⚠️ Could not map all ACC teams via API. Mapped:", len(acc_team_ids))

    # API game ids for ACC teams
    api_game_ids = set()
    for tid in acc_team_ids:
        for g in iter_games(client, "ncaamb", SEASON_ID, tid):
            gid = g.get("id")
            if gid:
                api_game_ids.add(gid)

    # DB game ids for ACC teams (by matchup)
    conn = sqlite3.connect("data/skout.db")
    cur = conn.cursor()
    acc_list = tuple(ACC_TEAMS)
    cur.execute(
        """
        SELECT game_id, home_team, away_team FROM games
        WHERE season_id = ? AND (home_team IN ({}) OR away_team IN ({}))
        """.format(
            ",".join(["?"] * len(acc_list)), ",".join(["?"] * len(acc_list))
        ),
        (SEASON_ID, *acc_list, *acc_list),
    )
    db_games = cur.fetchall()
    db_game_ids = {r[0] for r in db_games}

    missing_in_db = sorted(api_game_ids - db_game_ids)
    if missing_in_db:
        print(f"❌ Missing games in DB: {len(missing_in_db)}")
        for gid in missing_in_db[:20]:
            print(" -", gid)
        if len(missing_in_db) > 20:
            print(f" ... +{len(missing_in_db)-20} more")
    else:
        print("✅ All API-accessible ACC games are in DB.")

    # Event counts per game (API vs DB)
    cur.execute(
        """
        SELECT game_id, COUNT(play_id) FROM plays
        WHERE game_id IN ({})
        GROUP BY game_id
        """.format(",".join(["?"] * len(db_game_ids))),
        tuple(db_game_ids),
    )
    db_play_counts = {gid: cnt for gid, cnt in cur.fetchall()}

    mismatches = []
    for gid in sorted(api_game_ids):
        payload = client.get_game_events("ncaamb", gid)
        if not payload:
            continue
        events = [e for e in _unwrap_list_payload(payload) if isinstance(e, dict)]
        api_count = len(events)
        db_count = db_play_counts.get(gid, 0)
        if api_count != db_count:
            mismatches.append((gid, api_count, db_count))

    if mismatches:
        print(f"❌ Event count mismatches: {len(mismatches)}")
        for gid, api_c, db_c in mismatches[:20]:
            print(f" - {gid}: API={api_c} DB={db_c}")
        if len(mismatches) > 20:
            print(f" ... +{len(mismatches)-20} more")
    else:
        print("✅ All ACC game event counts match DB plays.")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
