from __future__ import annotations

import argparse
import re
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.ncaa_di_mens_basketball import NCAA_DI_MENS_BASKETBALL
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


def _parse_height(ht: str) -> int | None:
    m = re.match(r"(\d+)'(\d+)", ht)
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2))


def _load_lines(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # normalize spacing
    raw = raw.replace("\u2019", "'")
    return [l.strip() for l in raw.splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ACC roster height/weight/pos from text.")
    parser.add_argument("--file", required=True, help="Path to roster text file")
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

    updated = 0
    matched = 0
    current_team = None

    # Regex for player line: "Last, First ... 6'3" 185 G ..."
    player_re = re.compile(r"^([A-Za-z\'\-\.\s]+?),\s+([A-Za-z\'\-\.\s]+)\s+(\d+'\d+\")\s+(\d+)\s+([A-Z/]+)\b")

    for line in lines:
        # detect team header
        for header, team in TEAM_HEADERS.items():
            if line.startswith(header):
                current_team = team
                break
        if current_team is None:
            continue

        m = player_re.match(line)
        if not m:
            continue

        last, first, ht, wt, pos = m.groups()
        full_name = f"{first.strip()} {last.strip()}"
        key = (_norm_name(full_name), current_team)
        pid = name_team_to_id.get(key)
        if not pid:
            # fallback: ignore team mismatch by name only
            for (name_key, team_key), pid_val in name_team_to_id.items():
                if name_key == _norm_name(full_name):
                    pid = pid_val
                    break
        if not pid:
            continue

        matched += 1
        height_in = _parse_height(ht)
        weight_lb = int(wt)

        cur.execute(
            """
            UPDATE players
            SET height_in = COALESCE(?, height_in),
                weight_lb = COALESCE(?, weight_lb),
                position = COALESCE(?, position),
                team_id = COALESCE(?, team_id)
            WHERE player_id = ?
            """,
            (height_in, weight_lb, pos, current_team, pid),
        )
        if cur.rowcount:
            updated += 1

    conn.commit()
    conn.close()

    print(f"✅ Matched: {matched} | Updated: {updated}")


if __name__ == "__main__":
    main()
