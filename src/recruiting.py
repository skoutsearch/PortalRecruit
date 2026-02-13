from __future__ import annotations

from typing import Dict, List, Tuple

from src.archetypes import assign_archetypes
from src.valuation import estimate_nil_value


def _resolve_archetype(player_row: Dict) -> list[str]:
    if player_row.get("archetype"):
        return [str(player_row.get("archetype"))]
    badges = player_row.get("badges") or player_row.get("archetypes")
    if isinstance(badges, list) and badges:
        return [str(b) for b in badges]
    stats = {
        "ppg": player_row.get("ppg"),
        "rpg": player_row.get("rpg"),
        "apg": player_row.get("apg"),
        "height_in": player_row.get("height_in"),
        "weight_lb": player_row.get("weight_lb"),
        "shot3_percent": player_row.get("three_pt_pct") or player_row.get("shot3_percent"),
    }
    pos = player_row.get("position")
    return assign_archetypes(stats, "", pos) or []


def _match_pitch(player_row: Dict, team_needs: list[str]) -> Tuple[str, str]:
    needs = [n.lower() for n in (team_needs or [])]
    archetypes = [a.lower() for a in _resolve_archetype(player_row)]
    three_pt = player_row.get("three_pt_pct") or player_row.get("shot3_percent")
    ppg = float(player_row.get("ppg") or 0)
    rpg = float(player_row.get("rpg") or 0)
    blk = float(player_row.get("blk") or 0)

    if any("shoot" in n for n in needs) and ("sniper" in archetypes or (three_pt and float(three_pt) >= 0.36)):
        return (
            "Immediate green light from deep. We are missing exactly your shooting gravity.",
            "🎯 Targeted 'Shooting' Need",
        )

    if any("size" in n for n in needs) and ("rim protector" in archetypes or rpg >= 7 or blk >= 1.5):
        return (
            "Be our anchor. We have the guards, we just need the paint beast.",
            "🎯 Targeted 'Size' Need",
        )

    if "sniper" in archetypes:
        return (
            "Immediate green light from deep. We are missing exactly your shooting gravity.",
            "🎯 Targeted 'Shooting' Need",
        )

    if ppg >= 15:
        return (
            "We need a bucket getter and your production jumps off the tape.",
            "🎯 Targeted 'Scoring' Need",
        )

    return (
        "We love your game and think you can elevate our culture.",
        "🎯 Culture Fit",
    )


def get_dm_template(name: str, coach: str, hook: str, value_prop: str) -> str:
    return (
        f"Hey {name}, {coach} here. {hook} {value_prop} "
        "Let’s jump on a call."
    )


def get_official_visit_template(name: str, coach: str, hook: str, value_prop: str, usage: str, nil_value: str) -> str:
    return (
        f"Dear {name},\n\n"
        f"Coach {coach} here. {hook}\n\n"
        f"Fit & Usage Projection:\n{usage}\n\n"
        f"Value Proposition:\n{value_prop}\n\n"
        f"NIL Outlook: {nil_value}\n\n"
        "We’d love to host you for an official visit and walk through your role in our system."
    )


def generate_pitch(player_row: Dict, team_needs: list[str], tone: str = "dm", coach: str = "Coach") -> str:
    name = player_row.get("name") or player_row.get("player_name") or "there"
    hook, _reason = _match_pitch(player_row, team_needs)
    ppg = float(player_row.get("ppg") or 0)
    projected = ppg * 1.2
    nil_value = estimate_nil_value(player_row)
    if ppg > 0:
        value_prop = (
            f"We saw you averaged {ppg:.1f} last year. In our system that looks like {projected:.1f} a night. "
            f"With your NIL valuation of {nil_value}, we can make this worth your while."
        )
        usage = f"We project you at {projected:.1f} PPG with the green light to attack early."
    else:
        value_prop = (
            f"Your skill set translates immediately in our system. "
            f"With your NIL valuation of {nil_value}, we can make this worth your while."
        )
        usage = "We project a featured role tailored to your strengths."

    if tone == "official":
        return get_official_visit_template(name, coach, hook, value_prop, usage, nil_value)
    return get_dm_template(name, coach, hook, value_prop)


def get_pitch_reason(player_row: Dict, team_needs: list[str]) -> str:
    _, reason = _match_pitch(player_row, team_needs)
    return reason
