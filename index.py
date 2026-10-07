"""Step 2b - Compute an embedding for every chunk and store it in Chroma.

An embedding is a vector (384 numbers here) that represents the meaning
of a text: two texts about the same thing get close vectors. Chroma is
the database that stores those vectors and finds the nearest ones fast.

Input:  data/chunks.jsonl  (made by ingest.py)
Output: chroma_db/         (the vector database, rebuilt from scratch each run)
"""

import json
import time
from pathlib import Path

import chromadb

from rag import COLLECTION, DB_DIR, embed

CHUNKS_FILE = Path("data/chunks.jsonl")
BATCH_SIZE = 64


def main():
    records = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
    print(f"{len(records)} chunks to index")

    client = chromadb.PersistentClient(path=DB_DIR)
    # Start from an empty collection so re-running never creates duplicates
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    start = time.time()
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        collection.add(
            ids=[f"chunk-{i + j}" for j in range(len(batch))],
            embeddings=embed([r["text"] for r in batch]),
            documents=[r["text"] for r in batch],
            metadatas=[{"title": r["title"], "section": r["section"], "url": r["url"]}
                       for r in batch],
        )
        print(f"  {min(i + BATCH_SIZE, len(records))}/{len(records)}", end="\r")

    print(f"\nDone: {collection.count()} chunks indexed in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
