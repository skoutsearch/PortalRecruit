import os
from dotenv import load_dotenv

load_dotenv()

SYNERGY_API_KEY = os.getenv("SYNERGY_API_KEY")
BASE_URL = "https://api.sportradar.com/synergy/basketball"