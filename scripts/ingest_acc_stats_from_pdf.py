from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.ingestion.db import connect_db, ensure_schema


@dataclass
class PdfPlayerStat:
    gp: int | None = None
    points: int | None = None
    fg_made: int | None = None
    shot3_made: int | None = None
    ft_made: int | None = None
    reb: int | None = None
    ast: int | None = None
    stl: int | None = None
    blk: int | None = None
    minutes: float | None = None


SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm_name(name: str) -> str:
    name = (name or "").lower()
    name = re.sub(r"[\.'\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    parts = [p for p in name.split() if p and p not in SUFFIXES]
    return " ".join(parts)


def _to_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        if "." in str(val):
            return int(float(val))
        return int(val)
    except Exception:
        return None


def _to_float(val: str | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def iter_jsonl(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def merge_stat(target: PdfPlayerStat, row: dict) -> None:
    section = row.get("section") or ""
    section = str(section)

    if section == "Scoring":
        target.gp = target.gp or _to_int(row.get("g"))
        target.fg_made = target.fg_made or _to_int(row.get("fg"))
        target.shot3_made = target.shot3_made or _to_int(row.get("fg3"))
        target.ft_made = target.ft_made or _to_int(row.get("ft"))
        target.points = target.points or _to_int(row.get("pts"))
        return

    if section == "Rebounding":
        target.gp = target.gp or _to_int(row.get("g"))
        target.reb = target.reb or _to_int(row.get("total"))
        return

    if section == "Steals":
        target.gp = target.gp or _to_int(row.get("g"))
        target.stl = target.stl or _to_int(row.get("no"))
        return

    if section == "Blocked Shots":
        target.gp = target.gp or _to_int(row.get("g"))
        target.blk = target.blk or _to_int(row.get("blk"))
        return

    if section == "Assist/Turnover Ratio":
        target.gp = target.gp or _to_int(row.get("g"))
        target.ast = target.ast or _to_int(row.get("ast"))
        return

    if section == "Minutes Played":
        target.gp = target.gp or _to_int(row.get("g"))
        target.minutes = target.minutes or _to_float(row.get("min"))
        return

    if section == "3-Point FG Made":
        target.gp = target.gp or _to_int(row.get("g"))
        target.shot3_made = target.shot3_made or _to_int(row.get("fg3"))
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ACC player stats from parsed PDF JSONL into DB.")
    parser.add_argument(
        "--jsonl",
        nargs="*",
        default=[],
        help="JSONL files (default: data/acc_stats_from_pdf_*.jsonl)",
    )
    args = parser.parse_args()

    if args.jsonl:
        paths = [Path(p) for p in args.jsonl]
    else:
        paths = sorted(Path("data").glob("acc_stats_from_pdf_*.jsonl"))

    if not paths:
        raise RuntimeError("No JSONL inputs found.")

    conn = connect_db()
    ensure_schema(conn)
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT season_id FROM games WHERE season_id IS NOT NULL")
    seasons = [r[0] for r in cur.fetchall()]
    if not seasons:
        raise RuntimeError("No season_id found in games table.")
    season_id = seasons[0]

    cur.execute("SELECT player_id, full_name, team_id FROM players WHERE full_name IS NOT NULL")
    players = cur.fetchall()
    name_index: dict[str, list[tuple[str, str | None]]] = {}
    for pid, full_name, team_id in players:
        key = _norm_name(full_name)
        name_index.setdefault(key, []).append((pid, team_id))

    agg: dict[str, PdfPlayerStat] = {}
    unmatched = 0
    ambiguous = 0
    matched = 0

    for row in iter_jsonl(paths):
        player = row.get("player")
        if not player:
            continue
        key = _norm_name(str(player))
        options = name_index.get(key) or []
        if not options:
            unmatched += 1
            continue
        if len(options) > 1:
            ambiguous += 1
            continue
        pid, _team_id = options[0]
        matched += 1
        agg.setdefault(pid, PdfPlayerStat())
        merge_stat(agg[pid], row)

    inserted = 0
    updated = 0
    for pid, stats in agg.items():
        cur.execute(
            """
            SELECT gp, points, fg_made, shot3_made, ft_made, reb, ast, stl, blk, minutes
            FROM player_season_stats
            WHERE player_id = ? AND season_id = ?
            """,
            (pid, season_id),
        )
        existing = cur.fetchone()
        cur.execute("SELECT team_id FROM players WHERE player_id = ?", (pid,))
        team_id = (cur.fetchone() or [None])[0]

        def pick(existing_val, new_val):
            if existing_val is None:
                return new_val
            if isinstance(existing_val, (int, float)) and existing_val == 0:
                return new_val if new_val not in (None, 0) else existing_val
            return existing_val

        if existing:
            gp, points, fg_made, shot3_made, ft_made, reb, ast, stl, blk, minutes = existing
            gp = pick(gp, stats.gp)
            points = pick(points, stats.points)
            fg_made = pick(fg_made, stats.fg_made)
            shot3_made = pick(shot3_made, stats.shot3_made)
            ft_made = pick(ft_made, stats.ft_made)
            reb = pick(reb, stats.reb)
            ast = pick(ast, stats.ast)
            stl = pick(stl, stats.stl)
            blk = pick(blk, stats.blk)
            minutes = pick(minutes, stats.minutes)

            cur.execute(
                """
                UPDATE player_season_stats
                SET team_id = COALESCE(?, team_id),
                    gp = ?,
                    points = ?,
                    fg_made = ?,
                    shot3_made = ?,
                    ft_made = ?,
                    reb = ?,
                    ast = ?,
                    stl = ?,
                    blk = ?,
                    minutes = ?
                WHERE player_id = ? AND season_id = ?
                """,
                (team_id, gp, points, fg_made, shot3_made, ft_made, reb, ast, stl, blk, minutes, pid, season_id),
            )
            updated += 1
        else:
            cur.execute(
                """
                INSERT INTO player_season_stats (
                    player_id, season_id, team_id, gp, points, fg_made, shot3_made, ft_made,
                    reb, ast, stl, blk, minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    season_id,
                    team_id,
                    stats.gp,
                    stats.points,
                    stats.fg_made,
                    stats.shot3_made,
                    stats.ft_made,
                    stats.reb,
                    stats.ast,
                    stats.stl,
                    stats.blk,
                    stats.minutes,
                ),
            )
            inserted += 1

    conn.commit()
    conn.close()

    print("✅ ACC PDF ingest complete")
    print(f"   matched: {matched}")
    print(f"   unmatched: {unmatched}")
    print(f"   ambiguous: {ambiguous}")
    print(f"   inserted: {inserted} | updated: {updated}")


if __name__ == "__main__":
    main()
