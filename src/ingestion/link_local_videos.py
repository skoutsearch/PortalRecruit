import os
import sqlite3
import re

DB_PATH = os.path.join(os.getcwd(), "data/skout.db")
VIDEO_DIR = os.path.join(os.getcwd(), "data/video_clips")

# 🛠 CONFIGURATION: Map Filename Terms -> Official DB Names (Exact Matches from your list)
ALIAS_MAP = {
    "nc state": "NorthCarolinaState",
    "miami": "MiamiFL",      # "Miami" in ACC usually implies FL
    "miami (oh)": "MiamiOH", # Explicit distinction
    "loyola (md)": "LoyolaMD",
    "penn": "Pennsylvania",  # Handles "Penn" or "Pennsylvania"
    "ole miss": "Mississippi", # Just in case
    "uconn": "Connecticut",
    "st. john's": "St.John's", # Handle periods if needed
    "st. bonaventure": "StBonaventure",
    "saint mary's": "SaintMary's",
    "saint peter's": "SaintPeter's"
}

def clean_for_comparison(text):
    """
    Aggressive cleaning to match 'FloridaState' with 'Florida State'.
    Removes spaces, punctuation (except &), and lowers case.
    """
    # 1. Lowercase
    text = text.lower()
    # 2. Remove 'university', 'men's basketball', etc.
    text = text.replace("university", "").replace("men's basketball", "").replace("condensed", "")
    # 3. Remove spaces and common punctuation (keeping & for William & Mary)
    text = re.sub(r'[\s\.\-\(\)]', '', text) 
    return text

def check_match(filename_raw, db_name):
    """
    Checks if the DB name matches the filename, utilizing specific normalization.
    """
    # Normalize both sides to "floridastate" / "miamifl" style
    fname_norm = clean_for_comparison(filename_raw)
    db_norm = clean_for_comparison(db_name)

    # 1. Direct Match (Normalized)
    # e.g. db="FloridaState" -> norm="floridastate" matches file="Florida State" -> norm="floridastate"
    if db_norm in fname_norm:
        return True
        
    # 2. Check Aliases
    # If the filename contains a specific alias key (e.g., "nc state"), 
    # we accept it IF the db_name matches the alias's target.
    filename_lower = filename_raw.lower()
    for alias_key, official_db_name in ALIAS_MAP.items():
        if alias_key in filename_lower:
            # Does this alias map to the current DB team we are checking?
            if official_db_name == db_name:
                return True
            
    return False

def find_match(filename, db_cursor):
    # Fetch all games 
    db_cursor.execute("SELECT game_id, home_team, away_team, date FROM games")
    all_games = db_cursor.fetchall()
    
    best_match = None
    
    for game in all_games:
        g_id, home, away, date = game
        
        # We need BOTH teams to match
        home_match = check_match(filename, home)
        away_match = check_match(filename, away)
        
        if home_match and away_match:
            return g_id
            
    return None

def link_videos():
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ Video directory not found: {VIDEO_DIR}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.mkv', '.webm'))]
    print(f"📂 Scanning {len(files)} video files...")
    
    linked_count = 0
    
    for filename in files:
        game_id = find_match(filename, cursor)
        
        if game_id:
            full_path = os.path.abspath(os.path.join(VIDEO_DIR, filename))
            cursor.execute("UPDATE games SET video_path = ? WHERE game_id = ?", (full_path, game_id))
            print(f"🔗 MATCHED: '{filename}'")
            linked_count += 1
        else:
            print(f"⚠️  NO MATCH: '{filename}'")

    conn.commit()
    conn.close()
    print(f"\n✅ Total Videos Linked: {linked_count}/{len(files)}")

if __name__ == "__main__":
    link_videos()
