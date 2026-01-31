import chromadb
from src.processing.play_tagger import tag_play

DB_PATH = "data/vector_db"

client = chromadb.PersistentClient(path=DB_PATH)

clip_collection = client.get_or_create_collection(
    name="skout_video_clips"
)

def ingest_clip(
    clip_id: str,
    description: str,
    metadata: dict
):
    tags = tag_play(description, metadata.get("clock"))

    enriched_metadata = metadata | {
        "tags": tags
    }

    clip_collection.add(
        ids=[clip_id],
        documents=[description or "Basketball clip"],
        metadatas=[enriched_metadata]
    )
