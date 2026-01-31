import argparse
import os
import sys

import chromadb
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ingestion.synergy_client import SynergyClient

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

        season_list = []
        if isinstance(response, dict):
            season_list = response.get('data', [])
        elif isinstance(response, list):
            season_list = response
            
        print(f"DEBUG: Found {len(season_list)} total season records.")
        
        found_seasons = []

        for wrapper in season_list:
            season = wrapper.get("data", wrapper) if isinstance(wrapper, dict) else wrapper
            if not isinstance(season, dict):
                continue

            name = str(season.get('name', ''))
            year = str(season.get('year', '')) 
            sid = season.get('id')
            
            found_seasons.append({'name': name, 'id': sid, 'year': year})

            if str(target_year) in name or str(target_year) == year or str(target_year) == str(sid):
                print(f"✅ Found Matching Season: {name} (ID: {sid})")
                return sid
        
        if found_seasons:
            latest = found_seasons[0]
            print(f"⚠️ Defaulting to most recent available season: {latest['name']} (ID: {latest['id']})")
            return latest['id']

        print(f"❌ Season {target_year} not found in API response.")
        return None

    def fetch_all_teams(self, season_id):
        """Paginate through all teams to ensure we get everything."""
        all_teams = []
        skip = 0
        take = 500 # Maximize page size
        
        print(f"\n📥 Fetching ALL Teams for Season ID: {season_id}...")
        
        while True:
            # We need to modify client.get_teams to accept 'skip' if it doesn't already
            # For now, we manually assume the client handles the request, but we might need to patch client
            # if get_teams doesn't expose 'skip'. 
            # Looking at your client code, get_teams accepts params.
            
            # Manually constructing params here to override client defaults if needed
            params = {"seasonId": season_id, "take": take, "skip": skip}
            response = self.client._get("/ncaamb/teams", params=params)
            
            if not response:
                break
                
            # Extract data
            team_wrappers = []
            if isinstance(response, dict):
                team_wrappers = response.get('data', [])
            elif isinstance(response, list):
                team_wrappers = response
            
            if not team_wrappers:
                break
                
            for wrapper in team_wrappers:
                team = wrapper.get('data', wrapper) if isinstance(wrapper, dict) else wrapper
                if isinstance(team, dict) and team.get('id'):
                    all_teams.append(team)
            
            print(f"   - Fetched {len(team_wrappers)} teams (Total: {len(all_teams)})...")
            
            if len(team_wrappers) < take:
                break # End of list
            
            skip += take
            
        return all_teams

    def ingest_season_schedule(self, year):
        season_id = self.get_season_id(year)
        if not season_id:
            return

        teams = self.fetch_all_teams(season_id)
        
        if not teams:
            print("❌ No teams found. Check API permissions or Season ID.")
            return

        print(f"✅ Found {len(teams)} total teams. Starting Schedule Crawl...")
        
        processed_game_ids = set()
        
        for team in tqdm(teams, desc="Scanning Teams", unit="team"):
            team_id = team.get('id')
            if not team_id: continue

            # Pagination for games
            skip = 0
            take = 50 
            
            while True:
                # Manually call _get to support skip logic since get_games might hardcode params
                params = {"seasonId": season_id, "teamId": team_id, "take": take, "skip": skip}
                games_response = self.client._get("/ncaamb/games", params=params)
                
                if not games_response:
                    break
                
                game_wrappers = []
                if isinstance(games_response, dict):
                    game_wrappers = games_response.get('data', [])
                elif isinstance(games_response, list):
                    game_wrappers = games_response
                
                if not game_wrappers:
                    break

                for game_wrapper in game_wrappers:
                    game = game_wrapper.get('data', game_wrapper) if isinstance(game_wrapper, dict) else game_wrapper
                    
                    if not isinstance(game, dict): continue
                    
                    game_id = game.get('id')
                    if not game_id or game_id in processed_game_ids:
                        continue
                    
                    self.save_game_metadata(game, season_id)
                    processed_game_ids.add(game_id)
                
                if len(game_wrappers) < take:
                    break
                skip += take
                
        print(f"\n🎉 Ingestion Complete. {len(processed_game_ids)} unique games indexed.")

    def save_game_metadata(self, game_data, season_id):
        try:
            game_id = str(game_data.get('id'))
            
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
        except Exception:
            # Non-fatal: skip malformed items
            pass

    def pull_game_events(self, game_id):
        """Fetch events for a single game."""
        print(f"📥 Pulling Events for Game {game_id}...")
        response = self.client.get_game_events(league_code="ncaamb", game_id=game_id)

        events = []
        if response and isinstance(response, dict):
            event_wrappers = response.get("data", [])
            events = [
                ew.get("data", ew) for ew in event_wrappers if isinstance(ew, dict)
            ]
        elif isinstance(response, list):
            events = response

        if events:
            print(f"✅ Retrieved {len(events)} events.")
        return events

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
