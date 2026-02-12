import sys
import os
from pathlib import Path

# Fix path to allow imports
sys.path.append(str(Path.cwd()))

print("--- 1. Testing Search Logic (Hack 1) ---")
# We can't easily run Streamlit here, but we can check the file syntax
try:
    import Home
    print("✅ Home.py compiled successfully (Syntax check passed).")
except ImportError as e:
    # Expected if streamlit dependencies aren't fully mocked, but syntax is what we care about
    if "streamlit" in str(e):
        print("⚠️ Home.py requires Streamlit context, but syntax looks okay.")
    else:
        print(f"❌ Home.py Import Error: {e}")

print(" --- 2. Testing The Old Recruiter (Hack 2) ---")
try:
    from src.llm.scout import generate_scout_breakdown
    dummy_profile = {
        "name": "Test Rookie",
        "team_id": "Tech U",
        "height_in": 78,
        "weight_lb": 215,
        "stats": {"ppg": 22.4, "rpg": 8.0, "apg": 2.0, "gp": 30},
        "traits": {"dog_index": 90, "shot_making_index": 40},
        "plays": [
            (1, "drives and dunks hard", "g1", "10:00"),
            (2, "blocks shot into stands", "g1", "09:00"),
            (3, "fights for offensive board", "g2", "12:00")
        ]
    }
    report = generate_scout_breakdown(dummy_profile)
    print(f"✅ Report Generated ({len(report)} chars)")
    print(f"📝 Snippet: {report[:80]}...")
except Exception as e:
    print(f"❌ Scout Error: {e}")

print(" --- 3. Testing Reputation Check (Hack 3) ---")
try:
    from social_scout_worker import _check_reputation
    # Use a dummy query to test the API connection
    res = _check_reputation("John Doe", "Generic University")
    print(f"✅ Reputation Check executed. Result length: {len(res)}")
except Exception as e:
    print(f"❌ Social Worker Error: {e}")
