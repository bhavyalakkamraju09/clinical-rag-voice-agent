"""
Cross-encoder re-ranking: takes top-20 RRF candidates → returns top-5.
Model: ms-marco-MiniLM-L-6-v2 — precise (query, passage) joint scoring.
"""
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5


class CrossEncoderReranker:
    def __init__(self, model=None):
        self.model = model or CrossEncoder(RERANK_MODEL)

    def rerank(self, query: str, candidates: list[dict], top_k: int = TOP_K) -> list[dict]:
        """
        Adds rerank_score to each candidate and returns top_k sorted by it.
        """
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
