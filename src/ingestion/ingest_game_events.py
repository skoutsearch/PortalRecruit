import os
import requests
import sqlite3
import time
from dotenv import load_dotenv

load_dotenv("/media/jch903/fidelio/SKOUT/.env")

DB_PATH = os.path.join(os.getcwd(), "data/skout.db")
API_KEY = os.getenv("SYNERGY_API_KEY")

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # We drop the table to clear the "Unknown Play" junk data
    cursor.execute("DROP TABLE IF EXISTS plays")
    cursor.execute('''
        CREATE TABLE plays (
            play_id TEXT PRIMARY KEY,
            game_id TEXT,
            period INTEGER,
            clock_seconds INTEGER,
            clock_display TEXT,
            description TEXT,
            team_id TEXT,
            x_loc INTEGER,
            y_loc INTEGER,
            tags TEXT, 
            FOREIGN KEY(game_id) REFERENCES games(game_id)
        )
    ''')
    conn.commit()
    return conn

def get_linked_games():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT game_id, home_team, away_team FROM games WHERE video_path IS NOT NULL")
    return cursor.fetchall()

def retry_request(url, headers, max_retries=5):
    for attempt in range(max_retries):
        try:
            time.sleep(1.5)
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json().get('data', [])
            if response.status_code == 429:
                wait_time = (attempt + 1) * 5
                print(f"   ⏳ Rate limited. Cooling down for {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
        except Exception as e:
            print(f"   ❌ Network error: {e}")
            time.sleep(2)
    return []

def fetch_game_events(game_id):
    url = f"https://api.sportradar.com/synergy/basketball/ncaamb/games/{game_id}/events"
    headers = {"x-api-key": API_KEY}
    return retry_request(url, headers)

def process_events(events, game_id):
    parsed_plays = []
    for item in events:
        evt = item.get('data', {})
        
        # --- FIX: Use 'description' per Synergy Spec Page 41 ---
        # We also append flags like (PnR) if the API signals them explicitly
        desc_text = evt.get('description') or "Unknown Play"
        
        # Check explicit booleans in the spec (Page 40)
        extras = []
        if evt.get('pickAndRoll') is True:
            extras.append("(PnR)")
        if evt.get('transition') is True: # implied from tags usually, but checking key
            extras.append("(Trans)")
        
        if extras:
            desc_text += " " + " ".join(extras)
        
        # Clock handling
        raw_clock = evt.get('clock', 0)
        clock_sec = raw_clock if isinstance(raw_clock, int) else 0
        
        parsed_plays.append((
            evt.get('id'),
            game_id,
            evt.get('period'),
            clock_sec,
            str(raw_clock),
            desc_text,
            evt.get('offense', {}).get('id'),
            evt.get('shotX'),
            evt.get('shotY'),
            "" 
        ))
    return parsed_plays

def ingest_events():
    conn = setup_db() # This will DROP and RECREATE the table
    cursor = conn.cursor()
    
    games = get_linked_games()
    print(f"🎯 Re-Ingesting plays for {len(games)} games (fixing descriptions)...")
    
    total_new_plays = 0
    
    for game in games:
        g_id, home, away = game
        print(f"📥 Fetching: {home} vs {away}...")
        events = fetch_game_events(g_id)
        
        if events:
            rows = process_events(events, g_id)
            cursor.executemany('''
                INSERT OR REPLACE INTO plays 
                (play_id, game_id, period, clock_seconds, clock_display, description, team_id, x_loc, y_loc, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', rows)
            total_new_plays += len(rows)
            conn.commit()
            print(f"   ✅ Saved {len(rows)} plays.")
        else:
            print("   ⚠️ No events found.")

    conn.close()
    print(f"\n🚀 Repair Complete. {total_new_plays} valid plays stored.")

if __name__ == "__main__":
    ingest_events()
