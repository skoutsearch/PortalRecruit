from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.ingestion.db import connect_db, ensure_schema
from src.ingestion.synergy_client import SynergyClient


@dataclass(frozen=True)
class PipelinePlan:
    league_code: str
    season_id: str
    team_ids: list[str]  # empty => all accessible teams (if possible)
    ingest_events: bool = True


def _unwrap_list_payload(payload: Any) -> list[Any]:
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


def iter_games(
    client: SynergyClient,
    league_code: str,
    season_id: str,
    team_id: str | None,
    take: int = 100,
    max_pages: int = 50,
) -> Iterable[dict]:
    """Paginate through games using skip/take.

    NOTE: This relies on Synergy supporting `skip` for /games. If it doesn't,
    we'll still collect the first page.
    """

    skip = 0
    for _ in range(max_pages):
        payload = client.get_games(
            league_code=league_code,
            season_id=season_id,
            team_id=team_id,
            limit=take,
            skip=skip,
        )
        if not payload:
            break

        page = [g for g in _unwrap_list_payload(payload) if isinstance(g, dict)]
        if not page:
            break

        for g in page:
            yield g

        if len(page) < take:
            break
        skip += take


def upsert_games(conn, season_id: str, games: list[dict]) -> int:
    cur = conn.cursor()
    count = 0

    for game in games:
        status = game.get("status")
        if status not in {"GameOver", "Final", "Closed"}:
            continue

        home_team = (game.get("homeTeam") or {}).get("name", "Unknown")
        away_team = (game.get("awayTeam") or {}).get("name", "Unknown")

        cur.execute(
            """
            INSERT OR REPLACE INTO games
            (game_id, season_id, date, home_team, away_team, home_score, away_score, status, video_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT video_path FROM games WHERE game_id = ?), ?))
            """,
            (
                game.get("id"),
                season_id,
                game.get("date"),
                home_team,
                away_team,
                game.get("homeScore", 0),
                game.get("awayScore", 0),
                status,
                game.get("id"),
                None,
            ),
        )
        count += 1

    conn.commit()
    return count


def upsert_players(conn, team_id: str, players: list[dict]) -> int:
    cur = conn.cursor()
    rows = []

    for p in players:
        full_name = p.get("name") or p.get("fullName")
        if isinstance(full_name, dict):
            full_name = full_name.get("full") or full_name.get("fullName") or full_name
        if full_name is not None and not isinstance(full_name, str):
            try:
                import json
                full_name = json.dumps(full_name, ensure_ascii=False)
            except Exception:
                full_name = str(full_name)

        def safe_str(val):
            if val is None:
                return None
            if isinstance(val, str):
                return val
            try:
                import json
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)

        rows.append(
            (
                safe_str(p.get("id")),
                safe_str(team_id),
                safe_str(p.get("nameFirst") or p.get("firstName")),
                safe_str(p.get("nameLast") or p.get("lastName")),
                safe_str(full_name),
                safe_str(p.get("position")),
                p.get("heightInches") or p.get("height"),
                p.get("weightPounds") or p.get("weight"),
                safe_str(p.get("class") or p.get("classYear")),
            )
        )

    if not rows:
        return 0

    cur.executemany(
        """
        INSERT OR REPLACE INTO players
        (player_id, team_id, first_name, last_name, full_name, position, height_in, weight_lb, class_year)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def upsert_plays(conn, game_id: str, events: list[dict]) -> int:
    cur = conn.cursor()
    rows = []

    for evt in events:
        # description per Synergy spec
        desc_text = evt.get("description") or "Unknown Play"

        extras = []
        if evt.get("pickAndRoll") is True:
            extras.append("(PnR)")
        if evt.get("transition") is True:
            extras.append("(Trans)")
        if extras:
            desc_text += " " + " ".join(extras)

        # Synergy spec (Event schema):
        # - gameQuarter: int
        # - clock: int (int32). Treat as seconds remaining and format MM:SS.
        quarter = evt.get("gameQuarter")
        raw_clock = evt.get("clock")

        clock_sec = raw_clock if isinstance(raw_clock, int) else 0
        mm = max(0, clock_sec) // 60
        ss = max(0, clock_sec) % 60
        clock_display = f"{mm}:{ss:02d}"

        person = evt.get("person") or evt.get("player")
        player_id = None
        player_name = None
        if person:
            player_id = person.get("id")
            first = person.get("nameFirst") or ""
            last = person.get("nameLast") or ""
            player_name = f"{first} {last}".strip() or person.get("name")

        rows.append(
            (
                evt.get("id"),
                game_id,
                quarter if isinstance(quarter, int) else None,
                clock_sec,
                clock_display,
                desc_text,
                (evt.get("offense") or {}).get("id"),
                player_id,
                player_name,
                evt.get("shotX"),
                evt.get("shotY"),
                "",
            )
        )

    if not rows:
        return 0

    cur.executemany(
        """
        INSERT OR REPLACE INTO plays
        (play_id, game_id, period, clock_seconds, clock_display, description, team_id, player_id, player_name, x_loc, y_loc, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def run_pipeline(plan: PipelinePlan, api_key: str, progress_cb=None) -> dict:
    """Run a minimal end-to-end ingestion pipeline.

    progress_cb(step:str, info:dict) is optional.
    """

    client = SynergyClient(api_key=api_key)

    conn = connect_db()
    ensure_schema(conn)

    def tick(step: str, **info):
        if progress_cb:
            progress_cb(step, info)

    # 1) Games
    tick("schedule:start", season_id=plan.season_id)
    games: list[dict] = []

    if plan.team_ids:
        for tid in plan.team_ids:
            for g in iter_games(client, plan.league_code, plan.season_id, tid):
                games.append(g)
    else:
        for g in iter_games(client, plan.league_code, plan.season_id, None):
            games.append(g)

    inserted_games = upsert_games(conn, plan.season_id, games)
    tick("schedule:done", inserted_games=inserted_games)

    # 2) Players (if team_ids supplied)
    if plan.team_ids:
        for tid in plan.team_ids:
            payload = client.get_team_players(plan.league_code, tid)
            players = [p for p in _unwrap_list_payload(payload) if isinstance(p, dict)]
            upsert_players(conn, tid, players)

    # 3) Events
    inserted_plays = 0
    if plan.ingest_events:
        tick("events:start")
        cur = conn.cursor()
        cur.execute("SELECT game_id FROM games WHERE season_id = ?", (plan.season_id,))
        game_ids = [r[0] for r in cur.fetchall()]

        for idx, gid in enumerate(game_ids):
            if idx % 10 == 0:
                tick("events:progress", current=idx, total=len(game_ids))

            payload = client.get_game_events(plan.league_code, gid)
            if not payload:
                continue

            events = [e for e in _unwrap_list_payload(payload) if isinstance(e, dict)]
            inserted_plays += upsert_plays(conn, gid, events)

        tick("events:done", inserted_plays=inserted_plays)

    # 3) Derived Traits (optional)
    if inserted_plays > 0:
        try:
            from src.processing.derive_player_traits import build_player_traits
            build_player_traits()
        except Exception:
            pass

    conn.close()

    return {
        "inserted_games": inserted_games,
        "inserted_plays": inserted_plays,
    }
