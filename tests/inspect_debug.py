import os
import requests
import json
from dotenv import load_dotenv

# Load env vars
load_dotenv("/media/jch903/fidelio/SKOUT/.env")

SEASON_ID = "6085b5d0e6c2413bc4ba9122" # 2021-22
API_KEY = os.getenv("SYNERGY_API_KEY")

def inspect_data():
    url = "https://api.sportradar.com/synergy/basketball/ncaamb/games"
    headers = {"x-api-key": API_KEY}
    params = {
        "seasonId": SEASON_ID,
        "take": 50 # Just get 50 to check
    }

    print(f"📡 DEBUG: Fetching raw data for Season {SEASON_ID}...")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    games = data.get('data', [])
    print(f"📦 RAW COUNT: Received {len(games)} records from API.")

    if not games:
        print("⚠️  API returned an empty list. The Season ID might be wrong, or you strictly need a Competition ID.")
        return

    # 1. Check Statuses
    statuses = set(g.get('status') for g in games)
    print(f"\n🏷  STATUSES FOUND: {statuses}")

    # 2. Check First Record Keys
    first_game = games[0]
    print("\n🔎 SAMPLE GAME RECORD (Keys):")
    print(json.dumps(first_game, indent=2))

if __name__ == "__main__":
    inspect_data()
