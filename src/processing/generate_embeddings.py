import os
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DB_PATH = os.path.join(os.getcwd(), "data/skout.db")
VECTOR_DB_PATH = os.path.join(os.getcwd(), "data/vector_db")

def generate_embeddings():
    print("🧠 Loading AI Model (all-MiniLM-L6-v2)...")
    # This acts as a local, offline "Brain" for the system
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize Vector DB
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_or_create_collection(name="skout_plays")

    # Fetch Data
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT play_id, description, tags, game_id, clock_display FROM plays")
    rows = cursor.fetchall()
    conn.close()

    print(f"📦 Indexing {len(rows)} plays into Vector Database...")

    batch_size = 100
    for i in tqdm(range(0, len(rows), batch_size)):
        batch = rows[i:i+batch_size]
        
        ids = [r[0] for r in batch]
        
        # We combine description + tags to give the AI maximum context
        # e.g. "Missed 3pt Jump Shot (PnR) [Tags: 3pt, pnr, missed]"
        documents = [f"{r[1]} [Tags: {r[2]}]" for r in batch]
        
        metadatas = [
            {"game_id": r[3], "clock": r[4], "tags": r[2], "original_desc": r[1]} 
            for r in batch
        ]

        # Generate Embeddings
        embeddings = model.encode(documents).tolist()

        # Save to Chroma
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    print(f"✅ Successfully indexed {len(rows)} plays.")
    print("   The system is now ready for Semantic Search.")

if __name__ == "__main__":
    generate_embeddings()
