from __future__ import annotations

import os
import sqlite3
from typing import Dict

from src.film import analyze_tendencies, clean_clip_text

DB_PATH = os.path.join(os.getcwd(), "data/skout.db")

SYSTEM_PROFILES = {
    "5-Out Motion": {
        "Catch & Shoot": 0.4,
        "Cut": 0.3,
        "Passing": 0.3,
        "Post-Up": -0.5,
    },
    "Traditional": {
        "Post-Up": 0.5,
        "Passing": 0.2,
        "Catch & Shoot": 0.1,
        "Cut": 0.1,
    },
    "Triangle": {
        "Post-Up": 0.5,
        "Passing": 0.3,
        "Cut": 0.2,
    },
}


def _load_player_clips(player_id: str, limit: int = 80) -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT description FROM plays WHERE player_id = ? ORDER BY utc DESC LIMIT ?",
        (player_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [clean_clip_text(r[0]) for r in rows if r and r[0]]


def _tendency_vector(player_id: str) -> Dict[str, float]:
    clips = _load_player_clips(player_id)
    tendencies = analyze_tendencies(clips)
    totals = {k: float(v) for k, v in tendencies.items()}

    extra = {"Post-Up": 0, "Cut": 0, "Passing": 0}
    for clip in clips:
        txt = clip.lower()
        if "post-up" in txt or "post up" in txt or "left shoulder" in txt or "right shoulder" in txt:
            extra["Post-Up"] += 1
        if "cut" in txt:
            extra["Cut"] += 1
        if "assist" in txt or "pass" in txt:
            extra["Passing"] += 1

    for k, v in extra.items():
        if v > 0:
            totals[k] = totals.get(k, 0) + v

    total_actions = sum(totals.values()) or 1
    return {k: (v / total_actions) for k, v in totals.items()}


def _load_player_stats(player_id: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT height_in FROM players WHERE player_id = ? LIMIT 1", (player_id,))
    row = cur.fetchone()
    height_in = row[0] if row else None
    cur.execute(
        """
        SELECT shot3_attempt, fg_attempt, rpg
        FROM player_season_stats
        WHERE player_id = ?
        ORDER BY season_id DESC
        LIMIT 1
        """,
        (player_id,),
    )
    stats = cur.fetchone()
    conn.close()
    shot3_attempt = stats[0] if stats else None
    fg_attempt = stats[1] if stats else None
    rpg = stats[2] if stats else None
    return {
        "height_in": height_in,
        "shot3_attempt": shot3_attempt,
        "fg_attempt": fg_attempt,
        "rpg": rpg,
    }


def calculate_system_fit(player_id: str, system_profile: dict, system_name: str | None = None) -> float:
    vec = _tendency_vector(player_id)
    stats = _load_player_stats(player_id)
    score = 60.0

    for trait, weight in system_profile.items():
        score += weight * (vec.get(trait, 0.0) * 100)

    if system_name == "5-Out Motion":
        fg = stats.get("fg_attempt") or 0
        threes = stats.get("shot3_attempt") or 0
        three_rate = (threes / fg) if fg else 0
        height = stats.get("height_in") or 0
        if three_rate < 0.2:
            score -= 80
        if vec.get("Post-Up", 0) > 0.15:
            score -= 25
        if height and height >= 78:
            score -= 15
        rpg = stats.get("rpg") or 0
        if height and height >= 78 and rpg >= 7 and three_rate < 0.45:
            score = min(score, 35)
        if three_rate < 0.3 and ((height and height >= 78) or rpg >= 6 or vec.get("Post-Up", 0) > 0.05):
            score = min(score, 35)

    if system_name == "Traditional":
        height = stats.get("height_in") or 0
        rpg = stats.get("rpg") or 0
        post = vec.get("Post-Up", 0)
        if height and height >= 78:
            score += 20
        if rpg >= 6:
            score += 15
        if post >= 0.05:
            score += 10
        if height and height < 78:
            score -= 30

    if system_name == "Triangle":
        if vec.get("Post-Up", 0) < 0.12:
            score -= 20
        if vec.get("Passing", 0) < 0.08:
            score -= 15

    return max(0.0, min(100.0, score))


def grade_fit(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    return "D"
