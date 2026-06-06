"""
Reciprocal Rank Fusion (RRF) combining BM25 + FAISS ranked lists.
Parameter-free — no alpha tuning needed. k=60 is the standard constant.
"""
import json
from pathlib import Path

from .bm25_retriever import BM25Retriever
from .faiss_retriever import FAISSRetriever

K_RRF = 60
TOP_N  = 10
CHUNKS_PATH = Path("data/processed/chunks.json")


class HybridRetriever:
    def __init__(self):
        import json
        self.bm25 = BM25Retriever()
        self.faiss = FAISSRetriever()
        with open(CHUNKS_PATH) as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, top_n: int = TOP_N) -> list[dict]:
        """
        Returns top_n chunk dicts with added rrf_score.
        """
        bm25_results  = self.bm25.retrieve(query, top_n=top_n)
        faiss_results = self.faiss.retrieve(query, top_n=top_n)

        rrf: dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_results):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K_RRF + rank + 1)
        for rank, (idx, _) in enumerate(faiss_results):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K_RRF + rank + 1)

        sorted_ids = sorted(rrf, key=rrf.__getitem__, reverse=True)[:top_n]
        return [
            {**self.chunks[i], "rrf_score": rrf[i]}
            for i in sorted_ids
        ]
