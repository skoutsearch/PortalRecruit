import os
import sys
import argparse
import time
import chromadb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ingestion.synergy_client import SynergyClient


class GamePlayIngester:
    def __init__(self):
        self.client = SynergyClient()

        db_path = os.path.join(os.getcwd(), "data/vector_db")
        os.makedirs(db_path, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.play_collection = self.chroma_client.get_or_create_collection(
            name="skout_game_plays"
        )

    # --------------------------------------------------
    # FETCH PLAYS (LICENSE-SAFE)
    # --------------------------------------------------
    def fetch_game_plays(self, game_id):
        print(f"\n🔍 Fetching plays for game_id: {game_id}")

        skip = 0
        take = 200
        all_plays = []

        while True:
            response = self.client._get(
                f"/ncaamb/games/{game_id}/plays",
                params={"take": take, "skip": skip}
            )

            # ❗ Licensed endpoint may return None / 404
            if not response:
                print("⚠ No response returned from API — stopping play fetch")
                break

            plays = response.get("data", [])
            if not plays:
                break

            all_plays.extend(plays)

            if len(plays) < take:
                break

            skip += take
            time.sleep(0.3)

        print(f"✅ Retrieved {len(all_plays)} plays")
        return all_plays

    # --------------------------------------------------
    # INGEST
    # --------------------------------------------------
    def ingest_game(self, game_id):
        plays = self.fetch_game_plays(game_id)

        if not plays:
            print("❌ No plays found — nothing to ingest")
            return

        for wrapper in plays:
            play = wrapper.get("data", wrapper)
            self.save_play(play, game_id)

        print(f"🎉 Play ingestion complete for game {game_id}")

    # --------------------------------------------------
    # STORAGE
    # --------------------------------------------------
    def save_play(self, play, game_id):
        play_id = str(play.get("id"))
        if not play_id:
            return

        metadata = {
            "game_id": game_id,
            "event_type": play.get("eventType"),
            "description": play.get("description"),
            "clock": play.get("clock"),
            "period": play.get("period"),
            "sequence": play.get("sequence"),
        }

        document = play.get("description", "Unknown play")

        self.play_collection.add(
            ids=[play_id],
            documents=[document],
            metadatas=[metadata]
        )


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SKOUT – Game Play Ingestion (Synergy NCAA)"
    )
    parser.add_argument("--game-id", required=True)
    args = parser.parse_args()

    ingester = GamePlayIngester()
    ingester.ingest_game(args.game_id)
