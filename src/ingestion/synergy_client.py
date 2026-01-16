import requests
from config.settings import SYNERGY_API_KEY, BASE_URL

class SynergyClient:
    def __init__(self):
        self.headers = {"x-api-key": SYNERGY_API_KEY}

    def get_games(self, season_year):
        '''Fetch list of games for a season'''
        pass

    def get_play_by_play(self, game_id):
        '''Fetch events for a specific game'''
        pass