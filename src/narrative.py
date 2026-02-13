from __future__ import annotations

from typing import Dict, Any


def generate_physical_profile(
    player_name: str,
    position: str,
    height_pct: int,
    weight_pct: int,
    biometric_tags: list[str] | None,
    stats_dict: Dict[str, Any] | None,
    archetypes: list[str] | None = None,
) -> str:
    tags = biometric_tags or []
    tag_str = " / ".join(tags[:2]) if tags else "physical"
    pos_label = position or "Player"

    def _pct_label(pct: int) -> str:
        if pct > 90:
            return "Elite Size"
        if pct < 10:
            return "Undersized"
        return f"{pct}th %ile"

    ppg = (stats_dict or {}).get("ppg")
    rpg = (stats_dict or {}).get("rpg")
    apg = (stats_dict or {}).get("apg")

    def _fmt(stat, label):
        try:
            return f"{float(stat):.1f} {label}"
        except Exception:
            return None

    ppg_s = _fmt(ppg, "PPG")
    rpg_s = _fmt(rpg, "RPG")
    apg_s = _fmt(apg, "APG")

    arch_str = ""
    if archetypes:
        arch_str = " " + ", ".join(archetypes)

    if ppg_s or rpg_s or apg_s:
        stat_phrase = ", ".join([s for s in [ppg_s, rpg_s, apg_s] if s])
        return f"A {_pct_label(height_pct)} {pos_label} with {tag_str} traits producing {stat_phrase}.{arch_str}"

    return f"A {_pct_label(height_pct)} {pos_label} with {tag_str} traits.{arch_str}"
