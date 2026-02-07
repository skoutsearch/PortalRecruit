from __future__ import annotations

import re
from typing import Iterable

from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.ncaa_di_mens_basketball import NCAA_DI_MENS_BASKETBALL
from src.ingestion.db import connect_db, ensure_schema
from src.ingestion.synergy_client import SynergyClient


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _unwrap_list(payload) -> list[dict]:
    if not payload:
        return []
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        data = payload.get("data") or payload.get("items") or payload.get("result")
        if isinstance(data, dict):
            data = data.get("items") or data.get("data")
        if isinstance(data, list):
            return [p for p in data if isinstance(p, dict)]
    return []


def _pick_season_id(client: SynergyClient, league_code: str, target_name: str) -> str:
    payload = client.get_seasons(league_code)
    items = _unwrap_list(payload)
    # Sometimes seasons returns a single dict wrapped in {data: {...}}
    if not items and isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        items = [payload["data"]]
    for s in items:
        name = str(s.get("name") or "")
        if name == target_name:
            return str(s.get("id"))
    raise RuntimeError(f"Season {target_name} not found in API.")


def main() -> None:
    league = "ncaamb"
    season_name = "2021-2022"

    client = SynergyClient()
    season_id = _pick_season_id(client, league, season_name)
    print(f"✅ Using season {season_name} ({season_id})")

    teams_payload = client.get_teams(league, season_id)
    teams = _unwrap_list(teams_payload)
    if not teams:
        raise RuntimeError("No teams returned from API.")

    acc_names = NCAA_DI_MENS_BASKETBALL.get("ACC", [])
    acc_norm = {_norm(n) for n in acc_names}

    acc_team_ids = []
    acc_team_names = {}
    for t in teams:
        name = t.get("name") or t.get("fullName") or t.get("shortName")
        tid = t.get("id") or t.get("teamId")
        if not name or not tid:
            continue
        if _norm(name) in acc_norm:
            acc_team_ids.append(str(tid))
            acc_team_names[str(tid)] = str(name)

    if not acc_team_ids:
        raise RuntimeError("No ACC teams matched from API.")

    print(f"✅ ACC teams matched: {len(acc_team_ids)}")

    # Fetch players for ACC teams
    api_players: dict[str, dict] = {}
    for tid in acc_team_ids:
        payload = client.get_team_players(league, tid)
        items = _unwrap_list(payload)
        for item in items:
            # Some payloads wrap player in `data` or `player`
            rec = item.get("data") if isinstance(item, dict) and "data" in item else item
            if not isinstance(rec, dict):
                continue
            pid = rec.get("id") or rec.get("playerId")
            if not pid:
                continue
            api_players[str(pid)] = rec

    print(f"✅ Players pulled from API: {len(api_players)}")

    conn = connect_db()
    ensure_schema(conn)
    cur = conn.cursor()

    # Update heights/weights
    updated = 0
    for pid, rec in api_players.items():
        height = rec.get("height")
        weight = rec.get("weight")
        if height is None and weight is None:
            continue
        cur.execute(
            """
            UPDATE players
            SET height_in = COALESCE(?, height_in),
                weight_lb = COALESCE(?, weight_lb)
            WHERE player_id = ?
            """,
            (height, weight, pid),
        )
        if cur.rowcount:
            updated += 1

    # Remove ACC players not in API list
    api_player_ids = set(api_players.keys())
    cur.execute("SELECT player_id, team_id FROM players")
    to_delete = []
    acc_norm_db = {_norm(n) for n in acc_names}
    for pid, team_id in cur.fetchall():
        if not pid:
            continue
        team_id_str = str(team_id or "")
        if team_id_str in acc_team_ids or _norm(team_id_str) in acc_norm_db:
            if str(pid) not in api_player_ids:
                to_delete.append(str(pid))

    for pid in to_delete:
        cur.execute("DELETE FROM players WHERE player_id = ?", (pid,))
        cur.execute("DELETE FROM player_traits WHERE player_id = ?", (pid,))
        cur.execute("DELETE FROM player_season_stats WHERE player_id = ?", (pid,))

    conn.commit()
    conn.close()

    print(f"✅ Updated heights/weights for {updated} players")
    print(f"✅ Deleted {len(to_delete)} players not in API")


if __name__ == "__main__":
    main()
