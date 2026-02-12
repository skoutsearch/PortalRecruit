import json
import os
from typing import List, Dict, Any, Optional

import requests


POSITION_AVERAGES = {
    "PG": (74, 190),
    "SG": (76, 200),
    "SF": (79, 215),
    "PF": (82, 235),
    "C": (84, 250),
    "G": (75, 195),
    "F": (80, 220),
}


def _norm_position(pos: str) -> str:
    p = (pos or "").upper()
    if "PG" in p:
        return "PG"
    if "SG" in p:
        return "SG"
    if "SF" in p:
        return "SF"
    if "PF" in p:
        return "PF"
    if "C" in p:
        return "C"
    if "G" in p:
        return "G"
    if "F" in p:
        return "F"
    return "G"


def calculate_relative_size(height_in: Optional[float], weight_lb: Optional[float], position: str) -> List[str]:
    tags: List[str] = []
    if height_in is None and weight_lb is None:
        return tags

    pos_key = _norm_position(position)
    avg_h, avg_w = POSITION_AVERAGES.get(pos_key, (76, 200))

    if height_in is not None:
        if height_in > avg_h + 2:
            tags += ["tall", "big"]
        elif height_in < avg_h - 2:
            tags += ["undersized"]

    if weight_lb is not None:
        if weight_lb > avg_w + 15:
            tags += ["heavy", "strong"]
        elif weight_lb < avg_w - 15:
            tags += ["lanky", "skinny"]

    return list(dict.fromkeys(tags))


def analyze_physique(image_url: Optional[str]) -> Optional[Dict[str, Any]]:
    if not image_url:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        prompt = (
            "Analyze the basketball player in this image. Classify their build into ONE of these categories: "
            "[Lanky, Muscular, Heavy, Stocky, Athletic]. Also, estimate if they look 'Conditioned' or 'Soft'. "
            "Return JSON with keys: build, conditioning."
        )
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("OPENAI_MODEL") or "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 120,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        try:
            return json.loads(content)
        except Exception:
            return {"raw": content}
    except Exception:
        return None


def generate_biometric_tags(player_data: Dict[str, Any]) -> Dict[str, Any]:
    height_in = player_data.get("height_in")
    weight_lb = player_data.get("weight_lb")
    position = player_data.get("position") or ""
    image_url = player_data.get("image_url")

    math_tags = calculate_relative_size(height_in, weight_lb, position)
    vision = analyze_physique(image_url)

    tags = list(math_tags)
    if isinstance(vision, dict):
        build = vision.get("build") or vision.get("Build")
        conditioning = vision.get("conditioning") or vision.get("Conditioning")
        if build:
            tags.append(str(build).lower())
        if conditioning:
            tags.append(str(conditioning).lower())
    return {
        "math_tags": math_tags,
        "vision": vision,
        "tags": list(dict.fromkeys([t for t in tags if t])),
    }
