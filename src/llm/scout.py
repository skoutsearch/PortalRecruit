import os
import random
from typing import Optional, Dict, Any

# The user has confirmed OpenAI is installed and works well.
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Fallback templates if LLM fails
FALLBACK_TEMPLATES = [
    "Kid's got some tools, but I need to see more consistency. The numbers suggest {strength}, but the film needs to back it up.",
    "Solid rotational piece. Brings {strength} to the table. If he can clean up the mistakes, he plays.",
    "A bit of a project. The physicals are there ({height}), but the feel is still developing.",
    "Lunch pail guy. Shows up, does the work. {strength} is his ticket to minutes."
]

def _get_client():
    # 1. Safety check: Is the library installed?
    if OpenAI is None:
        return None

    # 2. Try Environment Variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 3. Try Streamlit Secrets (if env var is missing)
    if not api_key:
        try:
            import streamlit as st
            # Safely check secrets without crashing if they don't exist
            if hasattr(st, "secrets"):
                api_key = st.secrets.get("OPENAI_API_KEY")
        except ImportError:
            pass
        except Exception:
            pass
            
    if not api_key:
        return None
        
    return OpenAI(api_key=api_key)

def generate_scout_breakdown(profile: Dict[str, Any]) -> str:
    """
    Generates a "Old Recruiter" persona breakdown using the OpenAI API.
    """
    client = _get_client()
    
    # Extract Data Points
    name = profile.get("name", "Unknown Player")
    team = profile.get("team_id", "Unknown Team")
    
    # Stats formatting
    stats = profile.get("stats", {}) or {}
    
    # Explicitly fetching per-game stats (User Requested)
    # Using safe float conversion in case of None/String
    def safe_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    ppg = safe_float(stats.get("ppg"))
    rpg = safe_float(stats.get("rpg"))
    apg = safe_float(stats.get("apg"))
    gp = int(safe_float(stats.get("gp")))
    
    season = stats.get("season_label", "Current Season")
    
    # Physicals
    height = safe_float(profile.get("height_in"))
    weight = safe_float(profile.get("weight_lb"))
    
    ht_fmt = f"{int(height // 12)}'{int(height % 12)}\"" if height > 0 else "Unknown Ht"
    wt_fmt = f"{int(weight)} lbs" if weight > 0 else ""

    # Proprietary Indices (The "Secret Sauce")
    traits = profile.get("traits", {}) or {}
    dog_index = traits.get("dog_index", 50)
    menace_index = traits.get("menace_index", 50)
    rim_pressure = traits.get("rim_pressure_index", 50)
    shot_making = traits.get("shot_making_index", 50)
    
    # Identify standout traits for context
    standouts = []
    if dog_index > 75: standouts.append("elite motor")
    if menace_index > 75: standouts.append("defensive disruptor")
    if rim_pressure > 75: standouts.append("downhill force")
    if shot_making > 75: standouts.append("bucket getter")
    if dog_index < 30: standouts.append("needs to play harder")
    
    context_str = ", ".join(standouts) if standouts else "balanced skill set"

    # Play Tags (Evidence)
    plays = profile.get("plays", [])
    recent_actions = []
    for _, desc, _, _ in plays[:5]:
        recent_actions.append(f"- {desc}")
    recent_tape = "\n".join(recent_actions)

    system_prompt = (
        "You are 'The Old Recruiter', a 30-year veteran college basketball scout. "
        "You are grumpy, skeptical, but incredibly insightful. You value toughness, 'high motor', and efficiency over flashiness. "
        "You speak in short, punchy sentences. You use scouting jargon like 'glue guy', 'traffic cop', 'lunch pail', 'dog', 'spacing', 'gravity', 'length'. "
        "Never use enthusiastic marketing speak. Be realistic. "
        "Structure your response in Markdown with these headers: 'The Read', 'The Good', 'The Bad', 'Verdict'. "
        "Keep it under 200 words."
    )

    user_prompt = f"""
    Evaluate this prospect:
    Name: {name}
    School: {team}
    Size: {ht_fmt}, {wt_fmt}
    Season: {season} over {gp} games
    Stats: {ppg:.1f} PPG, {rpg:.1f} RPG, {apg:.1f} APG
    
    Proprietary Metrics (0-100 scale):
    - Dog Index (Hustle/Grit): {dog_index}
    - Menace Index (Defense): {menace_index}
    - Rim Pressure: {rim_pressure}
    - Shot Making: {shot_making}
    
    Context: Plays like a {context_str}.
    
    Recent Tape Notes:
    {recent_tape}
    """

    # Fallback if no client
    if not client:
        strength = standouts[0] if standouts else "general versatility"
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
            temperature=0.7,
            max_tokens=350
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        strength = standouts[0] if standouts else "skill set"
        template = random.choice(FALLBACK_TEMPLATES)
        return f"**Scout Unavailable:** {template.format(strength=strength, height=ht_fmt)}"
