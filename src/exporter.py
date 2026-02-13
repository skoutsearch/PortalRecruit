from __future__ import annotations

import csv
import io
from typing import Any, Dict, List


def _split_name(full_name: str) -> tuple[str, str]:
    parts = [p for p in (full_name or "").strip().split() if p]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _height_inches(val: Any) -> str:
    try:
        if isinstance(val, str) and "'" in val:
            feet, inches = val.replace("\"", "").split("'")
            return str(int(feet) * 12 + int(inches))
        return str(int(float(val)))
    except Exception:
        return ""


def generate_synergy_csv(roster_list: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["First Name", "Last Name", "Team", "Jersey", "Position", "Height", "Weight", "Class", "Notes"])
    for p in roster_list:
        name = p.get("name") or p.get("Player") or p.get("player_name") or ""
        first, last = _split_name(str(name))
        team = p.get("team") or p.get("Team") or p.get("team_id") or ""
        jersey = p.get("jersey") or p.get("Jersey") or ""
        pos = p.get("position") or p.get("Position") or ""
        height = _height_inches(p.get("height_in") or p.get("Height") or "")
        weight = p.get("weight_lb") or p.get("Weight") or ""
        clazz = p.get("class_year") or p.get("Class") or ""
        canonical = p.get("canonical_position") or ""
        tags = p.get("biometric_tags") or []
        tag_str = " | ".join([str(t).title() for t in tags][:3])
        from src.archetypes import assign_archetypes
        stats = {"ppg": p.get("ppg"), "rpg": p.get("rpg"), "apg": p.get("apg"), "weight_lb": p.get("weight_lb")}
        badges = assign_archetypes(stats, " ".join(tags), pos)
        badge_str = " ".join([f"[{b}]" for b in badges])
        notes = p.get("notes") or f"[AI: {canonical}] {badge_str} {tag_str}".strip()
        writer.writerow([first, last, team, jersey, pos, height, weight, clazz, notes])
    return output.getvalue()


def generate_text_report(roster_list: List[Dict[str, Any]]) -> str:
    lines = ["PortalRecruit Shortlist Report", "-" * 32]
    for p in roster_list:
        name = p.get("name") or p.get("Player") or p.get("player_name") or "Unknown"
        team = p.get("team") or p.get("Team") or p.get("team_id") or ""
        pos = p.get("position") or p.get("Position") or ""
        height = p.get("height_in") or p.get("Height") or ""
        weight = p.get("weight_lb") or p.get("Weight") or ""
        lines.append(f"{name} | {team} | {pos} | {height} in | {weight} lb")
    return "\n".join(lines)


def generate_team_packet(roster_list: List[Dict[str, Any]]) -> str:
    blocks = []
    for p in roster_list:
        name = p.get("name") or p.get("Player") or p.get("player_name") or "Unknown"
        team = p.get("team") or p.get("Team") or p.get("team_id") or ""
        pos = p.get("position") or p.get("Position") or ""
        height = p.get("height_in") or p.get("Height") or ""
        weight = p.get("weight_lb") or p.get("Weight") or ""
        canonical = p.get("canonical_position") or ""
        tags = p.get("biometric_tags") or []
        tag_str = ", ".join([str(t).title() for t in tags])
        notes = p.get("notes") or ""
        blocks.append(
            "\n".join([
                f"# {name}",
                f"**Team:** {team}",
                f"**Position:** {pos}",
                f"**Height/Weight:** {height} in / {weight} lb",
                f"**AI Position:** {canonical}",
                f"**Biometric Tags:** {tag_str}",
                f"**Scout Notes:** {notes}",
            ])
        )
    return "\n\n---\n\n".join(blocks)
