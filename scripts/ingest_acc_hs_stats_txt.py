from __future__ import annotations

import argparse
import re
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.db import connect_db, ensure_schema

TEAM_HEADERS = {
    "Boston College Eagles": "Boston College",
    "Clemson Tigers": "Clemson",
    "Duke Blue Devils": "Duke",
    "Florida State Seminoles": "Florida State",
    "Georgia Tech Yellow Jackets": "Georgia Tech",
    "Louisville Cardinals": "Louisville",
    "Miami Hurricanes": "Miami FL",
    "North Carolina Tar Heels": "North Carolina",
    "NC State Wolfpack": "NC State",
    "Notre Dame Fighting Irish": "Notre Dame",
    "Pittsburgh Panthers": "Pittsburgh",
    "Syracuse Orange": "Syracuse",
    "Virginia Cavaliers": "Virginia",
    "Virginia Tech Hokies": "Virginia Tech",
    "Wake Forest Demon Deacons": "Wake Forest",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm_name(name: str) -> str:
    name = (name or "").lower()
    name = re.sub(r"[\.'\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    parts = [p for p in name.split() if p and p not in SUFFIXES]
    return " ".join(parts)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _load_lines(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\u2019", "'")
    return [l.strip() for l in raw.splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ACC roster HS + PPG/RPG/APG from text.")
    parser.add_argument("--file", required=True, help="Path to roster text file")
    parser.add_argument("--season", default="2021-2022", help="Season id/name to store")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)

    lines = _load_lines(path)
    conn = connect_db()
    ensure_schema(conn)
    cur = conn.cursor()

    # Build player lookup by normalized name + team
    cur.execute("SELECT player_id, full_name, team_id FROM players")
    name_team_to_id: dict[tuple[str, str], str] = {}
    for pid, full_name, team_id in cur.fetchall():
        if not pid or not full_name:
            continue
        key = (_norm_name(full_name), str(team_id or ""))
        name_team_to_id[key] = str(pid)

    updated_players = 0
    updated_stats = 0
    inserted_players = 0
    inserted_stats = 0
    current_team = None

    player_re = re.compile(r"^([A-Za-z\'\-\.\s]+?),\s+([A-Za-z\'\-\.\s]+)\s+(.+)$")
    stats_re = re.compile(r"([0-9.]+)\s+Pts,\s+([0-9.]+)\s+Reb,\s+([0-9.]+)\s+Ast")

    for line in lines:
        for header, team in TEAM_HEADERS.items():
            if line.startswith(header):
                current_team = team
                break
        if current_team is None:
            continue

        m = player_re.match(line)
        if not m:
            continue

        last, first, rest = m.groups()
        full_name = f"{first.strip()} {last.strip()}"

        # Extract stats if present
        stats_match = stats_re.search(rest)
        if stats_match:
            ppg, rpg, apg = map(float, stats_match.groups())
            hs = rest[: stats_match.start()].strip()
        else:
            ppg = rpg = apg = 0.0
            hs = rest.strip()

        # Remove trailing parenthetical notes from high school
        hs = re.sub(r"\([^)]*\)$", "", hs).strip()

        key = (_norm_name(full_name), current_team)
        pid = name_team_to_id.get(key)
        if not pid:
            # fallback: ignore team mismatch by name only
            for (name_key, team_key), pid_val in name_team_to_id.items():
                if name_key == _norm_name(full_name):
                    pid = pid_val
                    break

        if not pid:
            pid = f"acc_roster_{_slug(current_team)}_{_slug(full_name)}"
            first_name = first.strip()
            last_name = last.strip()
            cur.execute(
                """
                INSERT OR IGNORE INTO players
                (player_id, team_id, first_name, last_name, full_name, high_school)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (pid, current_team, first_name, last_name, full_name, hs or None),
            )
            if cur.rowcount:
                inserted_players += 1
        else:
            cur.execute(
                """
                UPDATE players
                SET high_school = COALESCE(?, high_school),
                    team_id = COALESCE(?, team_id)
                WHERE player_id = ?
                """,
                (hs or None, current_team, pid),
            )
            if cur.rowcount:
                updated_players += 1

        # Upsert season stats
        cur.execute(
            """
            INSERT INTO player_season_stats (player_id, season_id, team_id, ppg, rpg, apg)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id, season_id) DO UPDATE SET
                team_id=excluded.team_id,
                ppg=excluded.ppg,
                rpg=excluded.rpg,
                apg=excluded.apg
            """,
            (pid, args.season, current_team, ppg, rpg, apg),
        )
        if cur.rowcount:
            inserted_stats += 1
        else:
            updated_stats += 1

    conn.commit()
    conn.close()

    print(
        "✅ Players inserted: {} | Players updated: {} | Season stats upserted: {}".format(
            inserted_players, updated_players, inserted_stats
        )
    )


if __name__ == "__main__":
    main()
