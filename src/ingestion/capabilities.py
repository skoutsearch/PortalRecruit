from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.ingestion.synergy_client import SynergyClient


@dataclass(frozen=True)
class Season:
    id: str
    name: str
    year: int | None = None


@dataclass(frozen=True)
class Team:
    id: str
    name: str
    conference: str | None = None


@dataclass
class CapabilityReport:
    league: str
    seasons: list[Season]
    teams_by_season: dict[str, list[Team]]

    # endpoint accessibility info
    seasons_accessible: bool
    teams_accessible: dict[str, bool]  # seasonId -> accessible
    games_accessible: dict[str, bool]  # seasonId -> accessible

    warnings: list[str]


def _unwrap_list_payload(payload: Any) -> list[Any]:
    """Synergy responses often wrap each item as {data: {...}} inside {data: [...]}"""
    if not payload:
        return []

    if isinstance(payload, dict):
        items = payload.get("data", [])
    elif isinstance(payload, list):
        items = payload
    else:
        return []

    out: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item.get("data", item))
        else:
            out.append(item)
    return out


def _as_season(obj: Any) -> Season | None:
    if not isinstance(obj, dict):
        return None
    sid = str(obj.get("id", "") or "")
    if not sid:
        return None
    name = str(obj.get("name", "") or "")
    year_raw = obj.get("year")
    year: int | None
    try:
        year = int(year_raw) if year_raw is not None and str(year_raw).strip() != "" else None
    except Exception:
        year = None
    return Season(id=sid, name=name, year=year)


def _as_team(obj: Any) -> Team | None:
    if not isinstance(obj, dict):
        return None
    tid = str(obj.get("id", "") or "")
    if not tid:
        return None
    name = str(obj.get("name", "") or "")

    # conference isn't always present; best-effort
    conf = obj.get("conference")
    if isinstance(conf, dict):
        conf_name = conf.get("name")
    else:
        conf_name = conf

    conference = str(conf_name) if conf_name else None
    return Team(id=tid, name=name, conference=conference)


def discover_capabilities(
    api_key: str,
    league_code: str = "ncaamb",
    max_seasons: int = 6,
    probe_teams: bool = True,
    probe_games: bool = True,
) -> CapabilityReport:
    """Discover what the provided API key can access.

    This is intentionally conservative:
    - Minimal calls (seasons + a couple probes)
    - Never raises; returns warnings + booleans
    """

    client = SynergyClient(api_key=api_key)
    warnings: list[str] = []

    seasons_payload = client.get_seasons(league_code=league_code)
    seasons_accessible = seasons_payload is not None
    seasons: list[Season] = []

    if seasons_accessible:
        for raw in _unwrap_list_payload(seasons_payload):
            s = _as_season(raw)
            if s:
                seasons.append(s)

        # Prefer most recent year first if present
        seasons.sort(key=lambda s: (s.year or 0, s.name), reverse=True)
        seasons = seasons[:max_seasons]
    else:
        code = client.last_status_code
        if code == 403:
            warnings.append("Seasons endpoint returned 403 (key likely has restricted discovery access).")
        elif code == 401:
            warnings.append("Seasons endpoint returned 401 (invalid API key or missing entitlements).")
        else:
            warnings.append("Could not fetch seasons list (network/format issue or restricted access).")

    teams_by_season: dict[str, list[Team]] = {}
    teams_accessible: dict[str, bool] = {}
    games_accessible: dict[str, bool] = {}

    for s in seasons:
        # Probe teams
        if probe_teams:
            teams_payload = client.get_teams(league_code=league_code, season_id=s.id)
            ok = teams_payload is not None
            teams_accessible[s.id] = ok
            if ok:
                teams: list[Team] = []
                for raw_team in _unwrap_list_payload(teams_payload):
                    t = _as_team(raw_team)
                    if t:
                        teams.append(t)
                teams.sort(key=lambda t: t.name)
                teams_by_season[s.id] = teams
            else:
                teams_by_season[s.id] = []
                if client.last_status_code == 403:
                    warnings.append(f"Teams endpoint forbidden for season {s.year or s.name}.")

        # Probe games (cheap)
        if probe_games:
            games_payload = client.get_games(league_code=league_code, season_id=s.id, team_id=None, limit=1)
            ok = games_payload is not None
            games_accessible[s.id] = ok
            if not ok and client.last_status_code == 403:
                warnings.append(f"Games endpoint forbidden for season {s.year or s.name}.")

    return CapabilityReport(
        league=league_code,
        seasons=seasons,
        teams_by_season=teams_by_season,
        seasons_accessible=seasons_accessible,
        teams_accessible=teams_accessible,
        games_accessible=games_accessible,
        warnings=warnings,
    )
