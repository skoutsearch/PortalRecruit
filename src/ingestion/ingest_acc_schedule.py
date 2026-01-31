import argparse
import os
import sys
import sqlite3
from dotenv import load_dotenv

# 1. Load Environment Variables
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", ".env"))
load_dotenv(ENV_PATH)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.ingestion.synergy_client import SynergyClient  # noqa: E402

# Defaults (only used if no discovery data is available)
DEFAULT_SEASON_ID = "6085b5d0e6c2413bc4ba9122"  # legacy guess: 2021-2022

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "skout.db")


def setup_db():
    """Ensures the games table exists."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            season_id TEXT,
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            status TEXT,
            video_path TEXT
        )
    ''')
    conn.commit()
    return conn

def _unwrap_list_payload(payload):
    """Synergy responses often wrap each item as {data: {...}} inside {data: [...]}"""
    if not payload:
        return []
    if isinstance(payload, dict):
        items = payload.get("data", [])
    elif isinstance(payload, list):
        items = payload
    else:
        return []

    out = []
    for item in items:
        if isinstance(item, dict):
            out.append(item.get("data", item))
        else:
            out.append(item)
    return out


def discover_accessible_seasons(client: SynergyClient, league_code: str = "ncaamb"):
    seasons_payload = client.get_seasons(league_code=league_code)
    seasons = [s for s in _unwrap_list_payload(seasons_payload) if isinstance(s, dict)]
    return seasons


def fetch_season_games(season_id: str, league_code: str = "ncaamb"):
    """Fetches games for a season. If the API key is limited, surface what it *can* access."""
    client = SynergyClient()

    print(f"📡 Fetching schedule for Season ID: {season_id}...")
    games_payload = client.get_games(league_code=league_code, season_id=season_id, team_id=None, limit=100)

    # If this returned None, SynergyClient already printed details.
    # Provide a helpful next step: list accessible seasons.
    if not games_payload:
        print("\nℹ️  Could not fetch games for that season with this API key.")
        print("    Attempting to discover accessible seasons...")
        seasons = discover_accessible_seasons(client, league_code=league_code)
        if seasons:
            # Print a compact summary
            print("✅ Discovered accessible seasons:")
            for s in seasons[:25]:
                sid = s.get("id", "")
                name = s.get("name", "")
                year = s.get("year", "")
                print(f"   - {year} {name} (seasonId={sid})")
            if len(seasons) > 25:
                print(f"   ... and {len(seasons) - 25} more")
        else:
            print("⚠️  No seasons discovered. This key may not have access to /seasons either.")
        return []

    games = []
    for game in _unwrap_list_payload(games_payload):
        if not isinstance(game, dict):
            continue

        status = game.get("status")
        if status not in ["GameOver", "Final", "Closed"]:
            continue

        home_team = game.get("homeTeam", {}).get("name", "Unknown")
        away_team = game.get("awayTeam", {}).get("name", "Unknown")

        games.append(
            (
                game.get("id"),
                season_id,
                game.get("date"),
                home_team,
                away_team,
                game.get("homeScore", 0),
                game.get("awayScore", 0),
                status,
                None,
            )
        )

    print(f"✅ Found {len(games)} valid games (sampled page).")
    print("ℹ️  Note: this script currently fetches a first page/sample of games.")
    print("    We'll upgrade it to full pagination + team-scoped crawling as part of the new pipeline UI.")
    return games

def save_schedule(games):
    conn = setup_db()
    cursor = conn.cursor()
    
    count = 0
    for game in games:
        # Use INSERT OR REPLACE to update stats if we re-run it
        cursor.execute('''
            INSERT OR REPLACE INTO games 
            (game_id, season_id, date, home_team, away_team, home_score, away_score, status, video_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT video_path FROM games WHERE game_id = ?), ?))
        ''', game[:8] + (game[0], None)) # This complex logic preserves existing video_path links if they exist
        count += 1
        
    conn.commit()
    conn.close()
    print(f"💾 Successfully cached {count} games into skout.db")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PortalRecruit – Schedule Ingestion")
    parser.add_argument("--season-id", type=str, default=DEFAULT_SEASON_ID)
    parser.add_argument("--league", type=str, default="ncaamb")
    args = parser.parse_args()

    games_list = fetch_season_games(args.season_id, league_code=args.league)
    if games_list:
        save_schedule(games_list)
