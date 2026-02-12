import os
import json
import time
import sys
import traceback
import sqlite3
import requests
from pathlib import Path
from datetime import datetime
import instaloader

# --- SETUP PATHS ---
REPO_ROOT = Path(__file__).resolve().parents[0]  # Assuming root, adjust if in src/workers
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Adjust import based on your project structure
try:
    from src.ingestion.db import connect_db, ensure_schema
except ImportError:
    # Fallback if running from root without package structure
    def connect_db():
        return sqlite3.connect(os.path.join(REPO_ROOT, "data", "skout.db"))

    def ensure_schema(conn):
        pass

# --- CONFIG ---
SERPER_API_KEY = os.getenv("SERPER_API_KEY") or "bd4c038827725838d7768562725539512395379a"  # Replace with your key if env missing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o"

# --- HELPERS ---

def _utc_now():
    return datetime.utcnow().isoformat()


def _serper_search(query: str, type: str = "search", num: int = 5) -> list:
    """Generic wrapper for Serper.dev API."""
    if not SERPER_API_KEY:
        print("⚠️ Missing SERPER_API_KEY")
        return []
    endpoint = "https://google.serper.dev/search" if type == "search" else "https://google.serper.dev/news"
    try:
        payload = json.dumps({"q": query, "num": num})
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        response = requests.post(endpoint, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if type == "news":
            return data.get("news", [])
        return data.get("organic", [])
    except Exception as e:
        print(f"❌ Serper Error ({query}): {e}")
        return []


# --- HACK 3: REPUTATION CHECK ---

def _check_reputation(player_name: str, school: str) -> str:
    """
    Searches for 'Red Flags' (arrests, suspensions, academic issues).
    Returns a summary string to feed into the LLM.
    """
    print(f"🕵️ Running Reputation Check for {player_name}...")

    # Aggressive query for bad news
    query = f'"{player_name}" {school} (arrested OR suspended OR ineligible OR dismissed OR "kicked off" OR "entering transfer portal") -site:instagram.com -site:twitter.com'
    results = _serper_search(query, type="search", num=4)

    if not results:
        return "No immediate red flags found in public news search."

    # Synthesize snippets
    evidence = []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        # Basic filter to avoid false positives (e.g., "dismissed rumors")
        evidence.append(f"- {title}: {snippet} ({link})")

    return " ".join(evidence)


def _find_instagram_url(player_name: str, school: str) -> str:
    """Finds the most likely Instagram URL using Google Search."""
    query = f'"{player_name}" {school} basketball site:instagram.com'
    results = _serper_search(query, num=3)
    for r in results:
        link = r.get("link", "")
        if "instagram.com/" in link and "/p/" not in link and "/reel/" not in link:
            # Clean URL (remove query params)
            return link.split("?")[0]
    return ""


def _scrape_instagram(url: str) -> dict:
    """
    Uses Instaloader to get public profile metadata.
    Robust against login requirements (fetches only public bio/stats).
    """
    if not url:
        return {}
    shortcode = url.rstrip("/").split("/")[-1]
    print(f"📸 Scraping Instagram: {shortcode}")
    L = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(L.context, shortcode)
        # Extract recent captions for "Vibe Check"
        posts_text = []
        for post in profile.get_posts():
            if len(posts_text) >= 5:
                break
            if post.caption:
                posts_text.append(f"Post ({post.date}): {post.caption[:150]}...")

        return {
            "handle": profile.username,
            "followers": profile.followers,
            "bio": profile.biography,
            "posts": posts_text,
            "verified": profile.is_verified,
        }
    except Exception as e:
        print(f"⚠️ Instaloader failed: {e}")
        return {"handle": shortcode, "error": str(e)}


def _analyze_social(player_name: str, social_data: dict, reputation_data: str) -> dict:
    """Uses OpenAI to generate the final 'Character Report'."""
    if not OPENAI_API_KEY:
        return {"summary": "AI Key Missing", "risk": "Unknown"}

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are an expert NCAA Background Investigator. "
        "Your job is to assess 'Recruiting Risk' based on social media behavior AND public news records. "
        "You are cynical and protective of the coaching staff. "
        "If you see arrests or suspensions in the Reputation Data, flag them as HIGH RISK immediately."
    )

    user_prompt = f"""
PLAYER: {player_name}

--- SOURCE 1: INSTAGRAM ---
Handle: {social_data.get('handle')}
Followers: {social_data.get('followers')}
Bio: {social_data.get('bio')}
Recent Posts: {json.dumps(social_data.get('posts', []))}

--- SOURCE 2: PUBLIC REPUTATION CHECK (NEWS) ---
{reputation_data}

--- MISSION ---
Produce a JSON report with:
1. verified_handle: str
2. confidence: int (0-100)
3. vibe_check: str (Short summary of personality)
4. red_flags: list[str] (List SPECIFIC issues from news or toxic posts)
5. green_flags: list[str] (Leadership, family, faith, grind)
6. recruiting_risk: str (Low, Moderate, High, Critical)
7. summary: str (2 sentences for the Head Coach)
8. recommendation: str (Proceed, Proceed with Caution, or Drop)
Return ONLY valid JSON.
"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return {"summary": "Analysis Failed", "error": str(e)}


# --- WORKER LOOP ---

def _process_job(job_id, player_id, conn):
    print(f" 🚀 Processing Job {job_id} for PlayerID {player_id}")
    cur = conn.cursor()

    # 1. Fetch Player Details
    # Try different ID formats
    cur.execute("SELECT full_name, team_id, high_school FROM players WHERE player_id = ?", (player_id,))
    row = cur.fetchone()
    if not row and str(player_id).isdigit():
        cur.execute("SELECT full_name, team_id, high_school FROM players WHERE player_id = ?", (int(player_id),))
        row = cur.fetchone()

    if not row:
        print("❌ Player not found in DB")
        cur.execute("UPDATE social_scout_queue SET status='error', last_error='Player not found' WHERE id=?", (job_id,))
        conn.commit()
        return

    name, school, hs = row
    print(f"👤 Target: {name} ({school})")

    try:
        # 2. Find Social
        ig_url = _find_instagram_url(name, school or hs)

        # 3. Scrape Social
        social_data = _scrape_instagram(ig_url) if ig_url else {}

        # 4. HACK 3: Check Reputation
        reputation_data = _check_reputation(name, school or hs)

        # 5. Analyze
        report = _analyze_social(name, social_data, reputation_data)

        # 6. Save
        cur.execute(
            """
            INSERT OR REPLACE INTO social_scout_reports (player_id, status, report_json, chosen_url, platform, handle, updated_at)
            VALUES (?, 'complete', ?, ?, 'instagram', ?, ?)
            """,
            (
                player_id,
                json.dumps(report),
                ig_url,
                social_data.get("handle"),
                _utc_now()
            )
        )

        # Update Queue
        cur.execute("UPDATE social_scout_queue SET status='done', finished_at=? WHERE id=?", (_utc_now(), job_id))
        conn.commit()
        print(f"✅ Job {job_id} Complete. Risk Level: {report.get('recruiting_risk')}")

    except Exception as e:
        traceback.print_exc()
        cur.execute("UPDATE social_scout_queue SET status='error', last_error=?, finished_at=? WHERE id=?", (str(e), _utc_now(), job_id))
        conn.commit()


def main():
    print("🤖 Social Scout Worker v2 (Reputation Aware) Started...")

    # Init DB Schema if needed
    try:
        conn = connect_db()

        # Ensure queue table exists
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS social_scout_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                status TEXT,
                requested_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                last_error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS social_scout_reports (
                player_id TEXT PRIMARY KEY,
                status TEXT,
                report_json TEXT,
                chosen_url TEXT,
                platform TEXT,
                handle TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ DB Init Warning: {e}")

    while True:
        try:
            conn = connect_db()
            cur = conn.cursor()

            # Find next job
            cur.execute("SELECT id, player_id FROM social_scout_queue WHERE status='queued' ORDER BY requested_at ASC LIMIT 1")
            job = cur.fetchone()

            if job:
                job_id, player_id = job

                # Mark running
                cur.execute("UPDATE social_scout_queue SET status='running', started_at=? WHERE id=?", (_utc_now(), job_id))
                conn.commit()

                _process_job(job_id, player_id, conn)
            else:
                time.sleep(5)

            conn.close()
        except KeyboardInterrupt:
            print("🛑 Stopping Worker.")
            break
        except Exception as e:
            print(f"❌ Main Loop Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
