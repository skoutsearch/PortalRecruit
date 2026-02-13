from __future__ import annotations

import os
from typing import Optional


def get_secret(secret_name: str):
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


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


FALLBACK = (
    "High-motor role player who makes winning plays: screens, box-outs, rotations, and quick extra passes. "
    "Thrives as a connective piece with smart decisions, strong effort, and low mistakes. "
    "Impacts the game without needing high usage—rebounds, defends, and keeps the offense flowing."
)


def _get_client():
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY") or get_secret("openai_api_key")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def generate_hypothetical_bio(query_text: str) -> str:
    query = (query_text or "").strip()
    if not query:
        return ""
    client = _get_client()
    if client is None:
        return FALLBACK

    system = (
        "You are an expert scout. Write a 3-sentence Ideal Scouting Report for a player described as: "
        "'{query}'. Focus on specific basketball traits, actions, and stats that would appear in their bio. "
        "Do not mention the player's name."
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL") or "gpt-4o",
        messages=[{"role": "system", "content": system.format(query=query)}],
        temperature=0.4,
        max_tokens=220,
    )
    return resp.choices[0].message.content.strip()


def generate_player_comp_bio(target_player_name: str) -> str:
    name = (target_player_name or "").strip()
    if not name:
        return ""
    client = _get_client()
    if client is None:
        return FALLBACK

    system = (
        "You are an expert scout. Describe the playing style, physical profile, and skill set of "
        "{name} as if they were an anonymous prospect. Focus on specific strengths and roles. "
        "Do not mention their name."
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL") or "gpt-4o",
        messages=[{"role": "system", "content": system.format(name=name)}],
        temperature=0.4,
        max_tokens=220,
    )
    return resp.choices[0].message.content.strip()
