"""
Build and persist FAISS + BM25 indexes from raw clinical notes.
Run: python -m src.ingestion.indexer
"""
import json
import pickle
import faiss
import numpy as np
from pathlib import Path

from .chunker import chunk_notes
from .embedder import get_embedder, embed_chunks
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data")
RAW_NOTES = DATA_DIR / "raw" / "clinical_notes.json"
PROCESSED_DIR = DATA_DIR / "processed"

FAISS_PATH = PROCESSED_DIR / "faiss.index"
BM25_PATH  = PROCESSED_DIR / "bm25.pkl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"


def build_index(notes_path: str | Path = RAW_NOTES):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading notes from {notes_path}…")
    with open(notes_path) as f:
        notes = json.load(f)

    print(f"Chunking {len(notes)} notes…")
    chunks = chunk_notes(notes)
    print(f"→ {len(chunks)} chunks")

    # ── BM25 ────────────────────────────────────────────────────
    print("Building BM25 index…")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  ✓ BM25 saved → {BM25_PATH}")

    # ── FAISS ───────────────────────────────────────────────────
    print("Embedding chunks (sentence-transformers)…")
    model = get_embedder()
    embeddings = embed_chunks(chunks, model)
    # Vectors are already L2-normalised; IndexFlatIP gives cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"  ✓ FAISS index ({index.ntotal} vectors) saved → {FAISS_PATH}")

    # ── Chunks metadata ─────────────────────────────────────────
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)
    print(f"  ✓ Chunks metadata saved → {CHUNKS_PATH}")
    print(f"\n✅ Indexing complete. {len(chunks)} chunks ready for retrieval.")


if __name__ == "__main__":
    build_index()
