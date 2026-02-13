from __future__ import annotations

from typing import Dict, Any, List


def assign_archetypes(stats_dict: Dict[str, Any] | None, text_desc: str | None) -> List[str]:
    stats = stats_dict or {}
    text = (text_desc or "").lower()
    badges: List[str] = []

    three_pct = stats.get("shot3_percent") or stats.get("3p%") or stats.get("three_pct")
    try:
        three_pct = float(three_pct) if three_pct is not None else None
    except Exception:
        three_pct = None

    apg = stats.get("apg")
    rpg = stats.get("rpg")
    height = stats.get("height_in") or stats.get("height")
    try:
        apg = float(apg) if apg is not None else None
    except Exception:
        apg = None
    try:
        rpg = float(rpg) if rpg is not None else None
    except Exception:
        rpg = None
    try:
        height = float(height) if height is not None else None
    except Exception:
        height = None

    if any(k in text for k in ["shooter", "3pt", "range"]) or (three_pct is not None and three_pct > 0.35):
        badges.append("🔫 Sniper")
    if (apg is not None and apg > 3.5) or any(k in text for k in ["point guard", "vision"]):
        badges.append("🧠 Floor General")
    if (rpg is not None and rpg > 8.0) or "rebound" in text:
        badges.append("🚜 Glass Cleaner")
    if any(k in text for k in ["defender", "defense", "steal"]):
        badges.append("🛡️ Lockdown")
    if height is not None and height > 82 and ("🔫 Sniper" in badges or "🧠 Floor General" in badges):
        badges.append("🦄 Unicorn")

    if not badges:
        if rpg is not None and rpg > 6.0:
            badges.append("🚜 Glass Cleaner")
        elif apg is not None and apg > 2.5:
            badges.append("🧠 Floor General")
        elif three_pct is not None and three_pct > 0.32:
            badges.append("🔫 Sniper")

    return badges
