"""FAISS dense retriever using all-MiniLM-L6-v2 embeddings."""
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

FAISS_PATH = Path("data/processed/faiss.index")
EMBED_MODEL = "all-MiniLM-L6-v2"


class FAISSRetriever:
    def __init__(self, index=None, model=None):
        self.index = index or faiss.read_index(str(FAISS_PATH))
        self.model = model or SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query: str, top_n: int = 20) -> list[tuple[int, float]]:
        """Returns list of (chunk_index, cosine_score) sorted descending."""
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, top_n)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
