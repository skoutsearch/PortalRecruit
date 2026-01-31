import os
import sys
import chromadb
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.processing.play_tagger import tag_play

# Try importing simple embedding model (SentenceTransformers)
# If not installed, we fallback to simple tagging only.
try:
    from sentence_transformers import SentenceTransformer
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠ sentence-transformers not found. Installing recommended for AI search.")
    print("  pip install sentence-transformers")

class PlayEnricher:
    def __init__(self):
        db_path = os.path.join(os.getcwd(), "data/vector_db")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # We work on the existing play collection
        self.collection = self.client.get_collection(name="skout_game_plays")
        
        # Load Model (Small, fast, local model for sports text)
        if HAS_ML:
            print("🔄 Loading AI Embedding Model (all-MiniLM-L6-v2)...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def enrich_all(self):
        print("\n🚀 Starting Play Enrichment (Tagging + Embeddings)...")
        
        # 1. Fetch all plays
        # Note: In production with millions of plays, use pagination (limit/offset)
        existing_data = self.collection.get()
        ids = existing_data['ids']
        documents = existing_data['documents']
        metadatas = existing_data['metadatas']
        
        if not ids:
            print("❌ No plays found in database. Run ingestion first.")
            return

        total = len(ids)
        print(f"📦 Found {total} plays to process.")

        # 2. Process Batch
        # We update in batches to be safe
        batch_size = 100
        
        for i in tqdm(range(0, total, batch_size), desc="Enriching Plays"):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            
            # Arrays for update
            new_embeddings = []
            updated_metas = []

            # A. Generate Embeddings (if ML available)
            if HAS_ML:
                # Embed the descriptions directly
                embeddings = self.model.encode(batch_docs).tolist()
                new_embeddings.extend(embeddings)

            # B. Generate Tags
            for j, doc in enumerate(batch_docs):
                current_meta = batch_metas[j]
                clock = current_meta.get("clock", None)
                
                # Run rule-based tagger
                tags = tag_play(doc, clock)
                
                # Update Metadata
                # Convert list to string for simple storage/filtering if needed, 
                # or keep specific keys. Chroma handles basic lists, but comma-string is safer for some UIs.
                current_meta["tags"] = ", ".join(tags)
                current_meta["is_enriched"] = True
                updated_metas.append(current_meta)

            # 3. Write Updates back to Chroma
            # valid args depend on what we have. If no ML, don't pass embeddings.
            update_args = {
                "ids": batch_ids,
                "metadatas": updated_metas
            }
            if HAS_ML:
                update_args["embeddings"] = new_embeddings

            self.collection.update(**update_args)

        print(f"\n✅ Successfully enriched {total} plays with AI & Tactics.")

if __name__ == "__main__":
    enricher = PlayEnricher()
    enricher.enrich_all()
