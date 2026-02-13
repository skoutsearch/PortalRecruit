from __future__ import annotations

from typing import Dict, Any, List


def assign_archetypes(stats_dict: Dict[str, Any] | None, text_desc: str | None, position: str | None = None) -> List[str]:
    stats = stats_dict or {}
    text = (text_desc or "").lower()
    pos = (position or "").upper()
    badges: List[str] = []

    three_pct = stats.get("shot3_percent") or stats.get("3p%") or stats.get("three_pct")
    try:
        three_pct = float(three_pct) if three_pct is not None else None
    except Exception:
        three_pct = None

    apg = stats.get("apg")
    rpg = stats.get("rpg")
    weight = stats.get("weight_lb") or stats.get("weight")
    try:
        apg = float(apg) if apg is not None else None
    except Exception:
        apg = None
    try:
        rpg = float(rpg) if rpg is not None else None
    except Exception:
        rpg = None
    try:
        weight = float(weight) if weight is not None else None
    except Exception:
        weight = None

    is_guard = "G" in pos and "F" not in pos and "C" not in pos
    is_big = any(k in pos for k in ["F", "C"])

    if (rpg is not None and rpg > 8.0) or (is_guard and rpg is not None and rpg > 5.0) or "rebound" in text:
        badges.append("🚜 Glass Cleaner")
    if (apg is not None and apg > 4.5) or ((not is_guard) and apg is not None and apg > 3.0) or any(k in text for k in ["point guard", "vision"]):
        badges.append("🧠 Floor General")
    if is_big and ("3pt" in text or (three_pct is not None and three_pct > 0.33)):
        badges.append("🔫 Stretch Big")
    if is_big and weight is not None and weight > 245:
        badges.append("🧱 Enforcer")
    if any(k in text for k in ["defender", "defense", "steal"]):
        badges.append("🛡️ Lockdown")

    return badges
