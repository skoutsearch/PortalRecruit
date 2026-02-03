import os
import time

import requests

from config.settings import BASE_URL


class SynergyClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("SYNERGY_API_KEY")
        if not self.api_key:
            raise ValueError("❌ ERROR: SYNERGY_API_KEY not found (env/secrets missing)")

        self.base_url = BASE_URL
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Introspection for callers (capabilities, UI, etc.)
        self.last_status_code: int | None = None
        self.last_error: str | None = None

    def _get(self, endpoint, params=None, retries=6):
        """Executes a GET request with strict rate-limit handling.

        Returns:
            Parsed JSON (dict/list) on success, else None.

        Side effects:
            Populates last_status_code and last_error for callers.
        """

        url = f"{self.base_url}{endpoint}"
        print(f"  > Requesting URL: {url} with params: {params}")
        self.last_status_code = None
        self.last_error = None

        # Simple global throttle (min delay between requests)
        min_interval_s = 1.0
        now = time.time()
        last = getattr(self, "_last_request_ts", 0.0)
        if now - last < min_interval_s:
            time.sleep(min_interval_s - (now - last))

        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                self._last_request_ts = time.time()
                self.last_status_code = response.status_code
                print(f"  < Status Code: {response.status_code}")

                # 1. Handle Rate Limiting (429)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and str(retry_after).isdigit():
                        wait_time = int(retry_after)
                    else:
                        # exponential backoff + jitter
                        wait_time = min(60, (2 ** attempt) * 2) + (0.5 * attempt)
                    print(f"      ⚠️ Rate limit hit (429). Pausing for {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                # 2. Handle Server Errors (5xx)
                if response.status_code >= 500:
                    wait_time = min(30, 2 + attempt * 2)
                    print(f"      ⚠️ Server Error ({response.status_code}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.HTTPError as e:
                # Non-retry for 401/403/404-ish (no point hammering)
                self.last_error = str(e)
                if response is not None:
                    self.last_status_code = response.status_code

                if response is not None and response.status_code in {401, 403, 404}:
                    print(f"❌ API Failed on {endpoint}: {e}")
                    return None

                if attempt == retries - 1:
                    print(f"❌ API Failed on {endpoint}: {e}")
                    return None

            except Exception as e:
                self.last_error = str(e)
                if attempt == retries - 1:
                    print(f"❌ API Failed on {endpoint}: {e}")
                    return None

        return None

    def get_seasons(self, league_code="ncaamb"):
        # Spec: GET /{league}/seasons
        return self._get(f"/{league_code}/seasons")

    def get_teams(self, league_code, season_id):
        # Spec: GET /{league}/teams
        # We limit 'take' to 500 to avoid timeouts/heavy rate limits
        params = {"seasonId": season_id, "take": 500} 
        return self._get(f"/{league_code}/teams", params=params)

    def get_games(self, league_code, season_id, team_id=None, limit=20, skip: int | None = None):
        # Spec: GET /{league}/games
        params = {
            "seasonId": season_id,
            "take": limit,
        }
        if skip is not None:
            params["skip"] = skip
        if team_id:
            params["teamId"] = team_id

        return self._get(f"/{league_code}/games", params=params)

    def get_game_events(self, league_code, game_id):
        # Spec: GET /{league}/games/{gameId}/events
        return self._get(f"/{league_code}/games/{game_id}/events")

    def get_team_players(self, league_code, team_id):
        # Spec: GET /{league}/teams/{teamId}/players
        return self._get(f"/{league_code}/teams/{team_id}/players")
