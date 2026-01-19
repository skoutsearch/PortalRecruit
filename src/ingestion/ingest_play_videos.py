import os
import sys
import time
import argparse
import chromadb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ingestion.synergy_client import SynergyClient


class PlayVideoIngester:
    def __init__(self):
        self.client = SynergyClient()

        db_path = os.path.join(os.getcwd(), "data/vector_db")
        os.makedirs(db_path, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=db_path)

        self.play_collection = self.chroma_client.get_or_create_collection(
            name="skout_game_plays"
        )

        self.video_collection = self.chroma_client.get_or_create_collection(
            name="skout_play_videos"
        )

    # --------------------------------------------------
    # FETCH VIDEO FOR A SINGLE PLAY
    # --------------------------------------------------
    def fetch_play_video(self, play_id):
        response = self.client._get(
            f"/ncaamb/plays/{play_id}/video"
        )

        # Licensed endpoint may return None / 404
        if not response:
            return []

        videos = response.get("data", [])
        return videos

    # --------------------------------------------------
    # INGEST ALL PLAY VIDEOS
    # --------------------------------------------------
    def ingest_all(self, limit=None):
        print("\n🎥 Linking play → video assets\n")

        play_ids = self.play_collection.get()["ids"]

        if limit:
            play_ids = play_ids[:limit]

        total_videos = 0

        for idx, play_id in enumerate(play_ids, 1):
            print(f"[{idx}/{len(play_ids)}] Processing play {play_id}")

            videos = self.fetch_play_video(play_id)

            if not videos:
                continue

            for wrapper in videos:
                video = wrapper.get("data", wrapper)
                self.save_video(video, play_id)
                total_videos += 1

            time.sleep(0.25)

        print(f"\n🎉 Video ingestion complete: {total_videos} videos linked")

    # --------------------------------------------------
    # STORAGE
    # --------------------------------------------------
    def save_video(self, video, play_id):
        video_id = str(video.get("id"))
        if not video_id:
            return

        metadata = {
            "play_id": play_id,
            "video_url": video.get("url"),
            "start_time": video.get("startTime"),
            "end_time": video.get("endTime"),
            "angle": video.get("angle"),
            "quality": video.get("quality"),
        }

        document = f"Video for play {play_id}"

        self.video_collection.add(
            ids=[video_id],
            documents=[document],
            metadatas=[metadata]
        )


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SKOUT – Play to Video Ingestion (Synergy NCAA)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of plays (for testing)",
        default=None
    )

    args = parser.parse_args()

    ingester = PlayVideoIngester()
    ingester.ingest_all(limit=args.limit)
