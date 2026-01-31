import os
import sys
import sqlite3

# Add project root to path so we can import the tagger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.processing.play_tagger import tag_play

DB_PATH = os.path.join(os.getcwd(), "data/skout.db")

def apply_tags():
    print("🧠 Starting Smart Tagging Process...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Fetch all plays that haven't been tagged yet (or all if you want to re-run)
    # We grab the ID, Description, and Clock
    cursor.execute("SELECT play_id, description, clock_seconds FROM plays")
    all_plays = cursor.fetchall()
    
    print(f"📦 Processing {len(all_plays)} plays...")
    
    updates = []
    tagged_count = 0
    
    for row in all_plays:
        p_id, desc, clock = row
        
        # Run the Tagger Logic
        # Ensure clock is an int (it might be None in DB)
        c_val = int(clock) if clock is not None and str(clock).isdigit() else None
        
        tags_list = tag_play(desc, c_val)
        
        if tags_list:
            # Convert list ['3pt', 'missed'] -> string "3pt, missed"
            tags_str = ", ".join(tags_list)
            updates.append((tags_str, p_id))
            tagged_count += 1

    # 2. Bulk Update
    if updates:
        print("💾 Saving tags to database...")
        cursor.executemany("UPDATE plays SET tags = ? WHERE play_id = ?", updates)
        conn.commit()
    
    conn.close()
    print(f"✅ Successfully tagged {tagged_count} plays.")
    print("   (Example: A play was labeled: '3pt, jumpshot, late_clock, missed')")

if __name__ == "__main__":
    apply_tags()
