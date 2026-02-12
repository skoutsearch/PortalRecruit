from __future__ import annotations

from typing import Dict, Any, Optional

from src.position_calibration import score_positions, topk


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _height_diff(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "Height data unavailable."
    diff = a - b
    if abs(diff) < 0.1:
        return "Same height."
    taller = "Player A" if diff > 0 else "Player B"
    return f"{taller} is {abs(diff):.1f} inches taller."


def _weight_diff(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "Weight data unavailable."
    diff = a - b
    if abs(diff) < 0.1:
        return "Same weight."
    heavier = "Player A" if diff > 0 else "Player B"
    return f"{heavier} is {abs(diff):.1f} lbs heavier."


def _stat_diff(label: str, a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return f"{label} data unavailable."
    diff = a - b
    if abs(diff) < 0.05:
        return f"Both are even in {label}."
    leader = "Player A" if diff > 0 else "Player B"
    return f"{leader} is +{abs(diff):.1f} {label} higher."


def compare_players(player_a: Dict[str, Any], player_b: Dict[str, Any], query: str = "Big Guard") -> Dict[str, Any]:
    a_height = _safe_float(player_a.get("height_in"))
    b_height = _safe_float(player_b.get("height_in"))
    a_weight = _safe_float(player_a.get("weight_lb"))
    b_weight = _safe_float(player_b.get("weight_lb"))

    a_ppg = _safe_float(player_a.get("ppg"))
    b_ppg = _safe_float(player_b.get("ppg"))
    a_rpg = _safe_float(player_a.get("rpg"))
    b_rpg = _safe_float(player_b.get("rpg"))
    a_apg = _safe_float(player_a.get("apg"))
    b_apg = _safe_float(player_b.get("apg"))

    fit_a = score_positions(query, a_height, a_weight)
    fit_b = score_positions(query, b_height, b_weight)
    top_a = topk(fit_a, k=1)
    top_b = topk(fit_b, k=1)

    a_score = top_a[0][1] if top_a else 0.0
    b_score = top_b[0][1] if top_b else 0.0
    a_conf = (a_score / max(fit_a.values()) * 100.0) if fit_a and max(fit_a.values()) else 0.0
    b_conf = (b_score / max(fit_b.values()) * 100.0) if fit_b and max(fit_b.values()) else 0.0
    fit_diff = "Player A" if a_conf > b_conf else "Player B"

    return {
        "height_diff": _height_diff(a_height, b_height),
        "weight_diff": _weight_diff(a_weight, b_weight),
        "ppg_diff": _stat_diff("PPG", a_ppg, b_ppg),
        "rpg_diff": _stat_diff("RPG", a_rpg, b_rpg),
        "apg_diff": _stat_diff("APG", a_apg, b_apg),
        "fit_diff": f"{fit_diff} matches '{query}' better ({a_conf:.0f}% vs {b_conf:.0f}%).",
        "fit_scores": {
            "player_a": a_conf,
            "player_b": b_conf,
        },
    }
