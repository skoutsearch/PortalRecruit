import os
import sqlite3
from pathlib import Path

import chromadb
from tqdm import tqdm

from src.search.semantic import get_embedder

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "skout.db"
VECTOR_DB_PATH = REPO_ROOT / "data" / "vector_db"


def generate_embeddings():
    print("🧠 Loading AI Model (all-MiniLM-L6-v2)...")
    # This acts as a local, offline "Brain" for the system
    model = get_embedder()

    # Initialize Vector DB
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    collection = client.get_or_create_collection(name="skout_plays")

    # Fetch Data
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT play_id, description, tags, game_id, clock_display, player_id, player_name FROM plays")
    rows = cursor.fetchall()
    conn.close()

    print(f"📦 Indexing {len(rows)} plays into Vector Database...")

    batch_size = 100
    for i in tqdm(range(0, len(rows), batch_size)):
        batch = rows[i:i + batch_size]

        ids = [r[0] for r in batch]

        # We combine description + tags + player hint for better retrieval quality.
        documents = [f"{r[6] or 'Unknown Player'} | {r[1]} [Tags: {r[2] or ''}]" for r in batch]

        metadatas = [
            {
                "game_id": r[3],
                "clock": r[4],
                "tags": r[2],
                "original_desc": r[1],
                "player_id": r[5],
                "player_name": r[6],
            }
            for r in batch
        ]

        # Generate Embeddings
        embeddings = model.encode(documents, normalize_embeddings=True).tolist()

        # Save to Chroma
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    print(f"✅ Successfully indexed {len(rows)} plays.")
    print("   The system is now ready for Semantic Search.")


if __name__ == "__main__":
    generate_embeddings()
