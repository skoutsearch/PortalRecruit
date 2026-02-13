from __future__ import annotations

import json
import os
from typing import Any, Dict, List

ROSTER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shortlist.json")
ROSTER_PATH = os.path.abspath(ROSTER_PATH)


def _load_roster() -> List[Dict[str, Any]]:
    if not os.path.exists(ROSTER_PATH):
        return []
    try:
        with open(ROSTER_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception:
        return []


def _save_roster(roster: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(ROSTER_PATH), exist_ok=True)
    with open(ROSTER_PATH, "w", encoding="utf-8") as f:
        json.dump(roster, f, indent=2)


def _dedupe_key(player: Dict[str, Any]) -> str:
    pid = str(player.get("player_id") or player.get("Player ID") or "").strip()
    if pid:
        return f"id:{pid}"
    name = str(player.get("name") or player.get("Player") or player.get("player_name") or "").strip().lower()
    return f"name:{name}" if name else ""


def add_player(player: Dict[str, Any]) -> bool:
    roster = _load_roster()
    key = _dedupe_key(player)
    if key and any(_dedupe_key(p) == key for p in roster):
        return False

    if "tier" not in player:
        player["tier"] = "Unranked"
    if "gm_notes" not in player:
        player["gm_notes"] = ""

    # enrich with biometrics + canonical position if missing
    if not player.get("biometric_tags") or not player.get("canonical_position"):
        from src.biometrics import generate_biometric_tags
        from src.position_calibration import score_positions, topk

        height = player.get("height_in")
        weight = player.get("weight_lb")
        position = player.get("position") or ""

        bio = generate_biometric_tags({
            "height_in": height,
            "weight_lb": weight,
            "position": position,
            "image_url": player.get("image_url"),
        })
        if not player.get("biometric_tags"):
            player["biometric_tags"] = bio.get("tags") or []

        if not player.get("canonical_position"):
            scores = score_positions(player.get("name") or "", height_in=height, weight_lb=weight)
            top = topk(scores, k=1)
            if top:
                player["canonical_position"] = top[0][0]

    roster.append(player)
    _save_roster(roster)
    return True


def update_player_tier(player_name: str, tier: str) -> bool:
    valid = ["S (Starter)", "A (Rotation)", "B (Deep Bench)", "C (Develop)", "F (Cut)", "Unranked"]
    if tier not in valid:
        return False
    roster = _load_roster()
    updated = False
    for p in roster:
        name = p.get("name") or p.get("Player") or p.get("player_name")
        if name == player_name:
            p["tier"] = tier
            updated = True
    if updated:
        _save_roster(roster)
    return updated


def update_player_notes(player_name: str, note: str) -> bool:
    roster = _load_roster()
    updated = False
    for p in roster:
        name = p.get("name") or p.get("Player") or p.get("player_name")
        if name == player_name:
            p["gm_notes"] = note
            updated = True
    if updated:
        _save_roster(roster)
    return updated


def remove_player(player_id: str) -> bool:
    roster = _load_roster()
    before = len(roster)
    roster = [p for p in roster if str(p.get("player_id") or p.get("Player ID") or "").strip() != str(player_id)]
    _save_roster(roster)
    return len(roster) < before


def get_roster() -> List[Dict[str, Any]]:
    return _load_roster()


def clear_roster() -> None:
    _save_roster([])
