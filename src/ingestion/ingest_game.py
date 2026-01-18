import os
import sys
import argparse
import time
from tqdm import tqdm
import chromadb

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ingestion.synergy_client import SynergyClient
from config.settings import SYNERGY_API_KEY

class GameIngester:
    def __init__(self):
        self.client = SynergyClient()
        
        # Initialize Vector DB for Metadata Storage
        db_path = os.path.join(os.getcwd(), "data/vector_db")
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # We use a separate collection for Game Metadata
        self.meta_collection = self.chroma_client.get_or_create_collection(name="skout_game_metadata")

    def get_season_id(self, target_year):
        """Finds the Synergy Season ID for a given year."""
        print(f"🔍 Searching for {target_year} season ID...")
        response = self.client.get_seasons(league_code="ncaamb")
        
        if not response:
            print("❌ Failed to fetch seasons.")
            return None

        # --- FIX: Handle Synergy API 'SeasonPaginationResponse' Structure ---
        # Structure: { "data": [ { "data": { "id": "...", "name": "..." } }, ... ] }
        
        season_list = []
        
        # 1. Extract the list of wrappers
        if isinstance(response, dict):
            season_list = response.get('data', [])
        elif isinstance(response, list):
            season_list = response
            
        print(f"DEBUG: Found {len(season_list)} total season records.")
        
        found_seasons = []

        for wrapper in season_list:
            # 2. Extract the actual season object from the wrapper
            season = wrapper.get('data', wrapper) if isinstance(wrapper, dict) else wrapper
            
            if not isinstance(season, dict):
                continue

            name = str(season.get('name', ''))
            year = str(season.get('year', '')) 
            sid = season.get('id')
            
            found_seasons.append({'name': name, 'id': sid, 'year': year})

            # Exact or partial match
            if str(target_year) in name or str(target_year) == year or str(target_year) == str(sid):
                print(f"✅ Found Matching Season: {name} (ID: {sid})")
                return sid
        
        # Debug output if no match found
        print("⚠️ Exact match not found. Available seasons:")
        for s in found_seasons[:5]: # Print first 5 to avoid spam
            print(f"   - {s['name']} (ID: {s['id']})")
            
        # Fallback: Return the most recent season (assuming sorted by ID or simply first in list)
        if found_seasons:
            latest = found_seasons[0]
            print(f"⚠️ Defaulting to most recent available season: {latest['name']} (ID: {latest['id']})")
            return latest['id']

        print(f"❌ Season {target_year} not found in API response.")
        return None

    def ingest_season_schedule(self, year):
        season_id = self.get_season_id(year)
        if not season_id:
            return

        print(f"\n📥 Fetching Teams for Season ID: {season_id}...")
        response = self.client.get_teams(league_code="ncaamb", season_id=season_id)
        
        if not response:
            print("❌ No teams found. Check API permissions or Season ID.")
            return

        # Handle TeamPaginationResponse
        team_wrappers = []
        if isinstance(response, dict):
            team_wrappers = response.get('data', [])
        elif isinstance(response, list):
            team_wrappers = response

        print(f"✅ Found {len(team_wrappers)} teams. Starting Schedule Crawl...")
        
        processed_game_ids = set()
        
        # Limit to first 50 teams for testing speed, remove [:50] for full run
        for team_wrapper in tqdm(team_wrappers[:50], desc="Scanning Teams", unit="team"):
            # Unwrap team data
            team = team_wrapper.get('data', team_wrapper) if isinstance(team_wrapper, dict) else team_wrapper
            
            if not isinstance(team, dict): continue
            
            team_id = team.get('id')
            if not team_id: continue

            # Fetch games for this team
            games_response = self.client.get_games(league_code="ncaamb", season_id=season_id, team_id=team_id, limit=50)
            
            if not games_response:
                continue
            
            game_wrappers = []
            if isinstance(games_response, dict):
                game_wrappers = games_response.get('data', [])
            elif isinstance(games_response, list):
                game_wrappers = games_response

            for game_wrapper in game_wrappers:
                # Unwrap game data
                game = game_wrapper.get('data', game_wrapper) if isinstance(game_wrapper, dict) else game_wrapper
                
                if not isinstance(game, dict): continue
                
                game_id = game.get('id')
                if not game_id or game_id in processed_game_ids:
                    continue
                
                self.save_game_metadata(game, season_id)
                processed_game_ids.add(game_id)
                
        print(f"\n🎉 Ingestion Complete. {len(processed_game_ids)} unique games indexed.")

    def save_game_metadata(self, game_data, season_id):
        try:
            game_id = str(game_data.get('id'))
            
            # Teams can be nested objects or just names depending on the endpoint view
            home_data = game_data.get('homeTeam', {})
            away_data = game_data.get('awayTeam', {})
            
            home_team = home_data.get('name', 'Unknown') if isinstance(home_data, dict) else "Unknown"
            away_team = away_data.get('name', 'Unknown') if isinstance(away_data, dict) else "Unknown"
            
            date = game_data.get('date', game_data.get('scheduled', 'Unknown Date'))
            description = f"{home_team} vs {away_team} on {date}"
            
            self.meta_collection.add(
                ids=[game_id],
                documents=[description],
                metadatas=[{
                    "season_id": str(season_id),
                    "game_date": str(date),
                    "home_team": str(home_team),
                    "away_team": str(away_team),
                    "status": str(game_data.get('status', 'unknown'))
                }]
            )
        except Exception as e:
            # print(f"Debug: metadata error {e}")
            pass

    def pull_game_events(self, game_id):
        print(f"📥 Pulling Events for Game {game_id}...")
        response = self.client.get_game_events(league_code="ncaamb", game_id=game_id)
        
        # EventPaginationResponse
        events = []
        if response and isinstance(response, dict):
             event_wrappers = response.get('data', [])
             events = [ew.get('data', ew) for ew in event_wrappers if isinstance(ew, dict)]
        elif isinstance(response, list):
            events = response

        if events:
            print(f"✅ Retrieved {len(events)} events.")
            return events
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SKOUT API Ingestion Engine")
    parser.add_argument("--year", type=int, help="Target Season Year (e.g. 2024)", default=2024)
    parser.add_argument("--game_id", type=str, help="Specific Game ID to pull events for")
    
    args = parser.parse_args()
    
    ingester = GameIngester()
    
    if args.game_id:
        ingester.pull_game_events(args.game_id)
    else:
        ingester.ingest_season_schedule(args.year)
