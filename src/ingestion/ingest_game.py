        for f in as_completed(futures):
            pid, path = futures[f]
            if f.result():
                extracted.append((pid, path))

    print(f"🎥 Extracted {len(extracted)} frames")

    for chunk in batch(extracted, BATCH_SIZE):
        ids, paths = zip(*chunk)
        vectors = [get_image_embedding(p) for p in paths]

        collection.add(
            ids=list(ids),
            embeddings=vectors,
            metadatas=[
                {
                    "filepath": p,
                    "game": f"{game['awayTeam']['abbr']} vs {game['homeTeam']['abbr']}",
                }
                for p in paths
            ],
        )

    print("🎉 Ingestion complete")


if __name__ == "__main__":
    run_ingestion()
