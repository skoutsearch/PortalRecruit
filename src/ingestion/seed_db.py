import sys
import os
import requests
import chromadb
import io
from PIL import Image

# Fix imports to allow running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.processing.vibe_check import get_image_embedding

# 1. Setup ChromaDB
db_path = os.path.join(os.getcwd(), "data/vector_db")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(name="skout_plays")

# 2. Define Test Images (Using "Bulletproof" GitHub Raw URLs)
# Note: These aren't basketball images, but they prove the AI works.
sample_data = [
    {
        "id": "play_001", 
        "desc": "High Energy Dog (Defense)", 
        "url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    },
    {
        "id": "play_002", 
        "desc": "Fast Traffic (Transition Offense)", 
        "url": "https://raw.githubusercontent.com/open-mmlab/mmdetection/master/demo/demo.jpg"
    },
    {
        "id": "play_003", 
        "desc": "Grocery Store (Zone Defense)", 
        "url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
    }
]

print("🚀 Starting Database Seed...")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SKOUT-Bot/1.0"
}

for item in sample_data:
    print(f"   Downloading: {item['desc']}...")
    
    try:
        response = requests.get(item['url'], headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        # Verify Image
        image_bytes = io.BytesIO(response.content)
        img_test = Image.open(image_bytes)
        img_test.verify() 

        # Save to disk
        filename = f"data/video_clips/{item['id']}.jpg"
        with open(filename, 'wb') as handler:
            handler.write(response.content)
            
        # Generate Vector
        print("   🧠 Generating AI Vector...")
        vector = get_image_embedding(filename)
        
        # Save to ChromaDB
        collection.add(
            ids=[item['id']],
            embeddings=[vector],
            metadatas=[{"description": item['desc'], "filepath": filename}]
        )
        print(f"   ✅ Saved {item['id']} to Database!")

    except Exception as e:
        print(f"   ❌ Error on {item['id']}: {e}")

print("\n🎉 Database Seeded! Run the dashboard now.")
