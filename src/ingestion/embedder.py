"""
Batch embedding of text chunks using sentence-transformers.
all-MiniLM-L6-v2: fast, free, good general + clinical semantic similarity.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def embed_chunks(chunks: list[dict], model: SentenceTransformer | None = None) -> np.ndarray:
    """Returns float32 numpy array of shape (n_chunks, embed_dim)."""
    if model is None:
        model = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine via IP on normalised vectors
    )
    return np.array(embeddings, dtype=np.float32)
