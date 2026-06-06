"""BM25 sparse retriever over clinical note chunks."""
import pickle
import numpy as np
from pathlib import Path

BM25_PATH   = Path("data/processed/bm25.pkl")
CHUNKS_PATH = Path("data/processed/chunks.json")


class BM25Retriever:
    def __init__(self, bm25=None, chunks=None):
        import json
        if bm25 is None:
            with open(BM25_PATH, "rb") as f:
                bm25 = pickle.load(f)
        if chunks is None:
            with open(CHUNKS_PATH) as f:
                chunks = json.load(f)
        self.bm25 = bm25
        self.chunks = chunks

    def retrieve(self, query: str, top_n: int = 20) -> list[tuple[int, float]]:
        """Returns list of (chunk_index, bm25_score) sorted descending."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked_ids = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in ranked_ids]
