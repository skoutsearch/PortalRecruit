from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
import instaloader

from src.ingestion.db import connect_db, ensure_schema

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o"


def _utc_now():
    return datetime.utcnow().isoformat()


def _build_social_query(player: dict) -> str:
    name = (player.get("name") or "").strip()
    school = (player.get("team_id") or "").strip()
    hs = (player.get("high_school") or "").strip()
    class_year = (player.get("class_year") or "").strip()
    class_term = f'"Class of {class_year}"' if class_year else ""

    parts = []
    if name:
        parts.append(f'"{name}"')
    parts.append("(site:instagram.com OR site:twitter.com OR site:x.com OR site:tiktok.com OR site:facebook.com)")

    validators = []
    if school:
        validators.append(f'"{school}"')
    if hs:
        validators.append(f'"{hs}"')
    if class_term:
        validators.append(class_term)
    validators.extend(["basketball", "athlete"])
    parts.append("(" + " OR ".join(validators) + ")")

    return " ".join([p for p in parts if p])


def _serper_search(query: str, num: int = 10) -> list[dict]:
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY missing")
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("organic", [])[:num]:
        results.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )
    return results


def _select_profile_via_llm(player: dict, results: list[dict]) -> str | None:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
You are a college basketball recruiting analyst. Identify the single URL that is most likely the player's PERSONAL social media profile.
Reject news, stats pages, highlight tapes, fan pages.
If not confident, return NOT_FOUND.

Player:
- Name: {player.get('name')}
- School: {player.get('team_id')}
- High School: {player.get('high_school')}
- Class: {player.get('class_year')}

Search Results JSON:
{json.dumps(results, ensure_ascii=False)}

Return ONLY the URL or NOT_FOUND.
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = (resp.choices[0].message.content or "").strip()
    if not choice or "NOT_FOUND" in choice.upper():
        return None
    return choice


def _scrape_instagram(url: str) -> dict:
    handle = url.split("instagram.com/")[-1].split("/")[0].strip()
    if not handle:
        raise RuntimeError("Could not parse Instagram handle")

    loader = instaloader.Instaloader()
    # If you have a session, load it here:
    # loader.load_session_from_file("burner_account")

    profile = instaloader.Profile.from_username(loader.context, handle)
    data = {
        "platform": "Instagram",
        "handle": handle,
        "bio": profile.biography,
        "followers": profile.followers,
        "posts": [],
    }
    count = 0
    for post in profile.get_posts():
        data["posts"].append(
            {
                "date": str(post.date_utc),
                "caption": post.caption,
                "likes": post.likes,
            }
        )
        count += 1
        if count >= 10:
            break
        time.sleep(1.5)
    return data


def _analyze_social(player: dict, social_data: dict) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = (
        "You are an NCAA lead recruiter and behavioral analyst."
        "Write a polished, human-sounding social media character assessment."
        "Focus on discipline, teamwork, leadership tone, work ethic, and any red/green flags."
    )
    user_prompt = f"""
PROSPECT:
Name: {player.get('name')}
School: {player.get('team_id')}
High School: {player.get('high_school')}
Class: {player.get('class_year')}

SOCIAL DATA:
Platform: {social_data.get('platform')}
Handle: {social_data.get('handle')}
Bio: {social_data.get('bio')}
Recent Posts: {json.dumps(social_data.get('posts', [])[:10], ensure_ascii=False)}

Return JSON with fields:
- verified_handle
- platform
- confidence (0-100)
- vibe_check (1-4 words)
- green_flags (list)
- red_flags (list)
- summary (4-7 sentences, well-crafted, human tone)
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _fetch_player(cur, player_id: str) -> dict:
    cur.execute(
        "SELECT player_id, full_name, team_id, class_year, high_school FROM players WHERE player_id = ?",
        (player_id,),
    )
    row = cur.fetchone()
    if not row:
        return {}
    return {
        "player_id": row[0],
        "name": row[1],
        "team_id": row[2],
        "class_year": row[3],
        "high_school": row[4],
    }


def _process_job(job_id: int, player_id: str) -> None:
    conn = connect_db()
    ensure_schema(conn)
    cur = conn.cursor()

    try:
        # mark in progress
        cur.execute(
            "UPDATE social_scout_queue SET status = 'running', started_at = ? WHERE id = ?",
            (_utc_now(), job_id),
        )
        conn.commit()

        player = _fetch_player(cur, player_id)
        if not player:
            raise RuntimeError("Player not found")

        query = _build_social_query(player)
        results = _serper_search(query, num=10)
        selected_url = _select_profile_via_llm(player, results)
        if not selected_url:
            raise RuntimeError("No profile found")

        social_data = None
        if "instagram.com" in selected_url:
            social_data = _scrape_instagram(selected_url)
        else:
            raise RuntimeError("Only Instagram scraping is supported in v1")

        report = _analyze_social(player, social_data)

        cur.execute(
            """
            INSERT INTO social_scout_reports
            (player_id, status, created_at, updated_at, search_query, search_results_json,
             chosen_url, platform, handle, bio, captions_json, report_json)
            VALUES (?, 'complete', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
              status='complete',
              updated_at=excluded.updated_at,
              search_query=excluded.search_query,
              search_results_json=excluded.search_results_json,
              chosen_url=excluded.chosen_url,
              platform=excluded.platform,
              handle=excluded.handle,
              bio=excluded.bio,
              captions_json=excluded.captions_json,
              report_json=excluded.report_json
            """,
            (
                player_id,
                _utc_now(),
                _utc_now(),
                query,
                json.dumps(results),
                selected_url,
                social_data.get("platform"),
                social_data.get("handle"),
                social_data.get("bio"),
                json.dumps(social_data.get("posts", [])),
                json.dumps(report),
            ),
        )

        cur.execute(
            "UPDATE social_scout_queue SET status='done', finished_at=?, last_error=NULL WHERE id = ?",
            (_utc_now(), job_id),
        )
        conn.commit()
    except Exception as e:
        cur.execute(
            "UPDATE social_scout_queue SET status='error', finished_at=?, last_error=? WHERE id = ?",
            (_utc_now(), str(e)[:500], job_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()


def main() -> None:
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is not set")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    print("✅ Social Scout Worker running...")
    while True:
        conn = connect_db()
        ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, player_id FROM social_scout_queue WHERE status = 'queued' ORDER BY requested_at ASC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()

        if not row:
            time.sleep(5)
            continue

        job_id, player_id = row
        try:
            _process_job(job_id, player_id)
        except Exception:
            traceback.print_exc()
            time.sleep(2)


if __name__ == "__main__":
    main()
