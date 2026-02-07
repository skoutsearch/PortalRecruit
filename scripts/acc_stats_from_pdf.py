from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pdfplumber


@dataclass
class SectionSpec:
    name: str
    columns: list[str]


SECTION_SPECS = {
    "Scoring": SectionSpec("Scoring", ["g", "fg", "fg3", "ft", "pts", "avg"]),
    "Rebounding": SectionSpec("Rebounding", ["g", "off", "def", "total", "avg"]),
    "Free Throw Percentage": SectionSpec("Free Throw Percentage", ["g", "ftm", "fta", "pct"]),
    "Steals": SectionSpec("Steals", ["g", "no", "avg"]),
    "3-Point FG Percentage": SectionSpec("3-Point FG Percentage", ["g", "fg3", "fga3", "pct"]),
    "3-Point FG Made": SectionSpec("3-Point FG Made", ["g", "fg3", "avg"]),
    "Blocked Shots": SectionSpec("Blocked Shots", ["g", "blk", "avg"]),
    "Assist/Turnover Ratio": SectionSpec("Assist/Turnover Ratio", ["g", "ast", "to", "ratio"]),
    "Minutes Played": SectionSpec("Minutes Played", ["g", "min", "avg"]),
}

TEAM_SECTION_SPECS = {
    "SCORING OFFENSE": SectionSpec("Scoring Offense", []),
    "SCORING DEFENSE": SectionSpec("Scoring Defense", []),
    "SCORING MARGIN": SectionSpec("Scoring Margin", []),
    "FREE THROW PERCENTAGE": SectionSpec("Free Throw Percentage (Team)", []),
    "FIELD GOAL PERCENTAGE": SectionSpec("Field Goal Percentage", []),
    "FIELD GOAL PCT DEFENSE": SectionSpec("Field Goal Pct Defense", []),
    "3-POINT FG PERCENTAGE": SectionSpec("3-Point FG Percentage (Team)", []),
    "3-POINT FG PCT DEFENSE": SectionSpec("3-Point FG Pct Defense", []),
    "TOTAL REBOUNDS PER GAME": SectionSpec("Total Rebounds Per Game", []),
    "OPPONENT TOTAL REBOUND": SectionSpec("Opponent Total Rebound", []),
    "REBOUNDING MARGIN": SectionSpec("Rebounding Margin", []),
    "BLOCKED SHOTS": SectionSpec("Blocked Shots (Team)", []),
    "ASSISTS PER GAME": SectionSpec("Assists Per Game", []),
    "STEALS PER GAME": SectionSpec("Steals Per Game", []),
    "TURNOVER MARGIN": SectionSpec("Turnover Margin", []),
    "ASSIST/TURNOVER RATIO": SectionSpec("Assist/Turnover Ratio (Team)", []),
    "OFFENSIVE REBOUNDS": SectionSpec("Offensive Rebounds", []),
    "DEFENSIVE REBOUNDS": SectionSpec("Defensive Rebounds", []),
    "DEFENSIVE REB PCT.": SectionSpec("Defensive Reb Pct", []),
    "OFFENSIVE REB PCT.": SectionSpec("Offensive Reb Pct", []),
    "3-POINT FG MADE PER GAME": SectionSpec("3-Point FG Made Per Game", []),
}

TEAM_PAIR_SECTIONS = {
    "SCORING OFFENSE": "SCORING DEFENSE",
    "SCORING MARGIN": "FREE THROW PERCENTAGE",
    "FIELD GOAL PERCENTAGE": "FIELD GOAL PCT DEFENSE",
    "3-POINT FG PERCENTAGE": "3-POINT FG PCT DEFENSE",
    "TOTAL REBOUNDS PER GAME": "OPPONENT TOTAL REBOUND",
    "ASSISTS PER GAME": "STEALS PER GAME",
    "TURNOVER MARGIN": "ASSIST/TURNOVER RATIO",
    "OFFENSIVE REBOUNDS": "DEFENSIVE REBOUNDS",
    "DEFENSIVE REB PCT.": "OFFENSIVE REB PCT.",
}

PAIR_SECTIONS = {
    "Scoring": "Rebounding",
    "Free Throw Percentage": "Steals",
    "3-Point FG Percentage": "3-Point FG Made",
    "Blocked Shots": "Assist/Turnover Ratio",
}

RANK_RE = re.compile(r"(\d+)\.\s+([^\-]+?)\s+-\s+([A-Za-z]+)\s+", re.MULTILINE)
TEAM_RANK_TOKEN = re.compile(r"(\d+)\.\s+")


def iter_entries(line: str) -> Iterable[tuple[str, str, str, str]]:
    """Yield (rank, player, team, trailing_text) for each ranked entry in line."""
    matches = list(RANK_RE.finditer(line))
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        trailing = line[start:end].strip()
        yield m.group(1), m.group(2).strip(), m.group(3).strip(), trailing


def parse_numbers(text: str) -> list[str]:
    return re.findall(r"\d+\.\d+|\d+", text)


def iter_team_entries(line: str) -> Iterable[tuple[int, int, str]]:
    matches = list(TEAM_RANK_TOKEN.finditer(line))
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        seg = line[start:end].strip()
        yield idx, int(m.group(1)), seg


def parse_team_segment(segment: str) -> tuple[str, list[str], str] | None:
    num_idx = None
    for i, ch in enumerate(segment):
        if ch.isdigit():
            num_idx = i
            break
    if num_idx is None:
        return None
    team = segment[:num_idx].strip()
    trailing = segment[num_idx:].strip()
    nums = parse_numbers(trailing)
    return team, nums, trailing


def normalize_header(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).upper()


def detect_sections(lines: list[str], section_specs: dict[str, SectionSpec]) -> list[tuple[str, list[str]]]:
    sections = []
    current_name = None
    buf: list[str] = []
    spec_keys = {normalize_header(k): k for k in section_specs.keys()}

    for line in lines:
        if not line.strip():
            continue

        header_match = None
        norm_line = normalize_header(line)
        for norm_key, raw_key in spec_keys.items():
            if norm_line.startswith(norm_key):
                header_match = raw_key
                break

        if header_match:
            if current_name and buf:
                sections.append((current_name, buf))
                buf = []
            current_name = header_match
            continue

        if current_name:
            buf.append(line)

    if current_name and buf:
        sections.append((current_name, buf))

    return sections


def parse_pdf(path: Path) -> list[dict]:
    records: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = [l.strip() for l in text.splitlines()]

            sections = detect_sections(lines, SECTION_SPECS)
            team_sections = detect_sections(lines, TEAM_SECTION_SPECS)

            for section_name, section_lines in sections:
                spec = SECTION_SPECS.get(section_name)
                if not spec:
                    continue

                paired = PAIR_SECTIONS.get(section_name)
                paired_spec = SECTION_SPECS.get(paired) if paired else None

                for line in section_lines:
                    if not RANK_RE.search(line):
                        continue

                    entries = list(iter_entries(line))
                    for entry_idx, (rank, player, team, trailing) in enumerate(entries):
                        active_section = section_name
                        active_spec = spec
                        if paired and entry_idx == 1:
                            active_section = paired
                            active_spec = paired_spec

                        nums = parse_numbers(trailing)
                        row = {
                            "section": active_section,
                            "rank": int(rank),
                            "player": player,
                            "team": team,
                            "values": nums,
                            "raw": trailing,
                        }
                        if active_spec and active_spec.columns and len(nums) >= len(active_spec.columns):
                            row.update({k: nums[i] for i, k in enumerate(active_spec.columns)})
                        records.append(row)

            for section_name, section_lines in team_sections:
                spec = TEAM_SECTION_SPECS.get(section_name)
                if not spec:
                    continue

                paired = TEAM_PAIR_SECTIONS.get(section_name)
                paired_spec = TEAM_SECTION_SPECS.get(paired) if paired else None

                for line in section_lines:
                    if not TEAM_RANK_TOKEN.search(line):
                        continue
                    for entry_idx, rank, segment in iter_team_entries(line):
                        parsed = parse_team_segment(segment)
                        if not parsed:
                            continue
                        team, nums, trailing = parsed

                        active_spec = spec
                        if paired and entry_idx == 1:
                            active_spec = paired_spec

                        if not active_spec:
                            continue
                        row = {
                            "section": active_spec.name,
                            "rank": rank,
                            "team": team,
                            "values": nums,
                            "raw": trailing,
                        }
                        records.append(row)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse ACC individual stats PDF into structured rows.")
    parser.add_argument("--pdf", required=True, help="Path to ACC stats PDF")
    parser.add_argument("--out", default="data/acc_stats_from_pdf.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    path = Path(args.pdf)
    if not path.exists():
        raise FileNotFoundError(path)

    records = parse_pdf(path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    main()
