import os
import random
from collections import Counter
from typing import Optional, Dict, Any


def get_secret(secret_name: str):
    """
    Tries to get secret from Snowflake secure storage first.
    Falls back to local st.secrets (for when you run locally).
    """
    try:
        from snowflake.snowpark.context import get_active_session  # noqa: F401
        import _snowflake
        return _snowflake.get_generic_secret_string(secret_name.upper())
    except Exception:
        try:
            import streamlit as st
            return st.secrets.get(secret_name.lower())
        except Exception:
            return None

# 1. SETUP & CLIENT
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Fallback templates if LLM fails or is missing
FALLBACK_TEMPLATES = [
    "Kid's got some tools, but I need to see more consistency. The numbers suggest {strength}, but the film needs to back it up.",
    "Solid rotational piece. Brings {strength} to the table. If he can clean up the mistakes, he plays.",
    "A bit of a project. The physicals are there ({height}), but the feel is still developing.",
    "Lunch pail guy. Shows up, does the work. {strength} is his ticket to minutes."
]

def _get_client():
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY") or get_secret("openai_api_key")
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                api_key = get_secret("openai_api_key")
        except:
            pass
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# 2. HELPER: FINGERPRINTING

def _generate_style_fingerprint(plays: list) -> str:
    """
    Analyzes ALL available plays to create a statistical profile of playstyle.
    Returns a string like: "Heavy reliance on spot-up (40%), active cutter (20%)."
    """
    if not plays:
        return "No film data available."

    # Simple keyword mapping if tags aren't pre-computed
    keywords = {
        "dunk": "Rim Finisher",
        "layup": "Rim Finisher",
        "3pt": "Shooter",
        "three": "Shooter",
        "jump shot": "Shooter",
        "assist": "Playmaker",
        "pass": "Playmaker",
        "rebound": "Glass Eater",
        "steal": "Disruptor",
        "block": "Rim Protector",
        "screen": "Screener",
        "post": "Post Threat",
        "drive": "Driver",
        "transition": "Transition"
    }

    counts = Counter()
    total_actions = 0

    for p in plays:
        # Handle tuple from DB or dict from JSON
        desc = p[1] if isinstance(p, (list, tuple)) and len(p) > 1 else str(p)
        desc_lower = desc.lower()
        matched = False
        for key, label in keywords.items():
            if key in desc_lower:
                counts[label] += 1
                matched = True
        if matched:
            total_actions += 1

    if total_actions == 0:
        return "Limited film sample."

    # Format the top 3 tendencies
    fingerprint = []
    for label, count in counts.most_common(3):
        pct = int((count / total_actions) * 100)
        fingerprint.append(f"{label} ({pct}%)")
    return ", ".join(fingerprint)

# 3. MAIN SCOUTING FUNCTION

def generate_scout_breakdown(profile: Dict[str, Any]) -> str:
    """
    Generates a 'cynical veteran scout' report based on stats, traits, and play-by-play data.
    """
    client = _get_client()

    # Extract Data
    name = profile.get("name", "Unknown")
    team = str(profile.get("team_id", "Unknown Team"))

    # Format Height/Weight
    ht = profile.get("height_in")
    wt = profile.get("weight_lb")
    if ht:
        ft = int(ht // 12)
        in_ = int(ht % 12)
        ht_fmt = f"{ft}'{in_}\""
    else:
        ht_fmt = "Height Unknown"

    wt_fmt = f"{int(wt)} lbs" if wt else "Weight Unknown"

    # Stats
    stats = profile.get("stats", {})
    ppg = stats.get("ppg", 0.0)
    rpg = stats.get("rpg", 0.0)
    apg = stats.get("apg", 0.0)
    gp = stats.get("gp", 0)
    season = stats.get("season_label", "Recent Season")

    # Traits (AI Indices)
    traits = profile.get("traits", {})
    dog_index = int(traits.get("dog_index", 50) or 50)
    menace_index = int(traits.get("menace_index", 50) or 50)
    shot_making = int(traits.get("shot_making_index", 50) or 50)
    rim_pressure = int(traits.get("rim_pressure_index", 50) or 50)

    # Film Room (The "Meat")
    plays = profile.get("plays", [])

    # HACK 2: GENERATE FINGERPRINT
    style_fingerprint = _generate_style_fingerprint(plays)

    # Get specific recent clips (Limit to 5 for the prompt text)
    recent_tape_lines = []
    for i, p in enumerate(plays[:5]):
        desc = p[1] if isinstance(p, (list, tuple)) else str(p)
        clock = p[3] if isinstance(p, (list, tuple)) and len(p) > 3 else ""
        recent_tape_lines.append(f"- {clock} {desc}")
    recent_tape = " ".join(recent_tape_lines)

    # Determine "Standout" for fallbacks
    standouts = []
    if ppg > 15:
        standouts.append("scoring punch")
    if rpg > 8:
        standouts.append("rebounding")
    if apg > 5:
        standouts.append("vision")
    if dog_index > 75:
        standouts.append("high motor")
    if menace_index > 75:
        standouts.append("defensive upside")
    strength = standouts[0] if standouts else "versatility"

    # SYSTEM PROMPT: THE "OLD RECRUITER" PERSONA
    system_prompt = (
        "You are 'The Old Recruiter,' a cynical, 30-year veteran basketball scout. "
        "You don't trust stats, you trust what you see. You speak in short, punchy sentences. "
        "You use scout jargon correctly (motor, length, spacing, gravity, downhill, heavy feet, twitchy). "
        "You are NOT a generic AI assistant. Do not use phrases like 'The player showcases' or 'In conclusion.' "
        "Be direct. Highlight the specific elite skill if it exists, otherwise call out the flaws. "
        "Keep the output under 120 words. Focus on: DOES HE TRANSLATE?"
    )

    position = profile.get("position", "") or "Unknown"
    class_year = profile.get("class_year", "") or ""

    user_prompt = f"""
SCOUTING REPORT REQUEST:
Prospect: {name} ({team})
Position/Year: {position} {class_year}
Frame: {ht_fmt}, {wt_fmt}
Production: {ppg:.1f} Pts, {rpg:.1f} Reb, {apg:.1f} Ast ({gp} games)

AI METRICS (0-100):
- Dog/Motor: {dog_index}
- Defense: {menace_index}
- Shooting: {shot_making}
- Rim Pressure: {rim_pressure}

PLAY STYLE FINGERPRINT (Film Frequency):
{style_fingerprint}

RECENT FILM NOTES (use these actions explicitly):
{recent_tape}

TASK:
Write a 3-4 sentence scouting blurb.
1. Identify his ARCHETYPE immediately based on the Fingerprint.
2. Comment on how his frame matches that archetype.
3. Use the 'Dog Index' to judge his motor.
4. Cite at least one specific action from the Recent Film Notes (e.g., "Left P&R", "Pick and Pops").
"""

    # FALLBACK LOGIC
    if not client:
        template = random.choice(FALLBACK_TEMPLATES)
        reason = "Library missing" if OpenAI is None else "Key missing"
        return f"**Scout Unavailable ({reason}):** {template.format(strength=strength, height=ht_fmt)}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,  # Lower temp for more grounded/cynical take
            max_tokens=250
        )
        content = response.choices[0].message.content.strip()

        # Format bolding for key phrases if not present
        if "**" not in content:
            # Heuristic: Bold the first 3 words
            parts = content.split(" ", 3)
            if len(parts) > 3:
                content = f"**{parts[0]} {parts[1]} {parts[2]}** {parts[3]}"

        return content

    except Exception as e:
        # Fallback on crash
        template = random.choice(FALLBACK_TEMPLATES)
        return f"**Scout Offline:** {template.format(strength=strength, height=ht_fmt)}"
