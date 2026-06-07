"""
Retrieval benchmark: BM25-only vs FAISS-only vs Hybrid (RRF).
Run: python evaluation/benchmark_retrieval.py
Outputs a comparison table and saves to evaluation/results/benchmark.json
"""
import sys, json, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_PATH  = Path("evaluation/golden_test_set.json")
RESULTS_DIR  = Path("evaluation/results")
RESULTS_DIR.mkdir(exist_ok=True)


def precision_at_k(retrieved_texts: list[str], expected_answer: str, k: int = 5) -> float:
    """Rough precision: fraction of top-k chunks that contain keywords from expected answer."""
    keywords = set(expected_answer.lower().split()) - {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and", "or"
    }
    if not keywords:
        return 0.0
    hits = sum(
        1 for text in retrieved_texts[:k]
        if any(kw in text.lower() for kw in keywords)
    )
    return hits / k


def run_benchmark():
    if not GOLDEN_PATH.exists():
        print("Golden test set not found. Run: make golden")
        return

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.faiss_retriever import FAISSRetriever
    from src.retrieval.rrf_fusion import HybridRetriever
    import json as _json

    with open("data/processed/chunks.json") as f:
        chunks = _json.load(f)

    bm25_retriever   = BM25Retriever()
    faiss_retriever  = FAISSRetriever()
    hybrid_retriever = HybridRetriever()

    results = {"bm25": [], "faiss": [], "hybrid": []}
    latencies = {"bm25": [], "faiss": [], "hybrid": []}

    print(f"\n{'─'*65}")
    print(f"{'Question':<40} {'BM25':>6} {'FAISS':>6} {'Hybrid':>7}")
    print(f"{'─'*65}")

    for item in golden[:30]:  # sample 30 for speed
        q = item["question"]
        expected = item["expected_answer"]

        # BM25
        t0 = time.time()
        bm25_ids = bm25_retriever.retrieve(q, top_n=10)
        bm25_texts = [chunks[i]["text"] for i, _ in bm25_ids if i < len(chunks)]
        latencies["bm25"].append((time.time()-t0)*1000)
        p_bm25 = precision_at_k(bm25_texts, expected)
        results["bm25"].append(p_bm25)

        # FAISS
        t0 = time.time()
        faiss_ids = faiss_retriever.retrieve(q, top_n=10)
        faiss_texts = [chunks[i]["text"] for i, _ in faiss_ids if i < len(chunks)]
        latencies["faiss"].append((time.time()-t0)*1000)
        p_faiss = precision_at_k(faiss_texts, expected)
        results["faiss"].append(p_faiss)

        # Hybrid
        t0 = time.time()
        hybrid_docs = hybrid_retriever.retrieve(q, top_n=10)
        hybrid_texts = [d["text"] for d in hybrid_docs]
        latencies["hybrid"].append((time.time()-t0)*1000)
        p_hybrid = precision_at_k(hybrid_texts, expected)
        results["hybrid"].append(p_hybrid)

        print(f"{q[:40]:<40} {p_bm25:>6.2f} {p_faiss:>6.2f} {p_hybrid:>7.2f}")

    # Summary
    avg_bm25   = sum(results["bm25"])   / len(results["bm25"])
    avg_faiss  = sum(results["faiss"])  / len(results["faiss"])
    avg_hybrid = sum(results["hybrid"]) / len(results["hybrid"])

    lat_bm25   = sum(latencies["bm25"])   / len(latencies["bm25"])
    lat_faiss  = sum(latencies["faiss"])  / len(latencies["faiss"])
    lat_hybrid = sum(latencies["hybrid"]) / len(latencies["hybrid"])

    hybrid_vs_bm25  = ((avg_hybrid - avg_bm25)  / max(avg_bm25,  0.001)) * 100
    hybrid_vs_faiss = ((avg_hybrid - avg_faiss) / max(avg_faiss, 0.001)) * 100

    print(f"\n{'═'*65}")
    print(f"  RETRIEVAL BENCHMARK RESULTS (P@5, n=30 questions)")
    print(f"{'═'*65}")
    print(f"  BM25-only:    P@5={avg_bm25:.3f}  latency={lat_bm25:.1f}ms")
    print(f"  FAISS-only:   P@5={avg_faiss:.3f}  latency={lat_faiss:.1f}ms")
    print(f"  Hybrid (RRF): P@5={avg_hybrid:.3f}  latency={lat_hybrid:.1f}ms")
    print(f"\n  Hybrid vs BM25:  {hybrid_vs_bm25:+.1f}%")
    print(f"  Hybrid vs FAISS: {hybrid_vs_faiss:+.1f}%")
    print(f"{'═'*65}")

    # Save
    benchmark = {
        "bm25":   {"p_at_5": round(avg_bm25, 3),   "avg_latency_ms": round(lat_bm25, 1)},
        "faiss":  {"p_at_5": round(avg_faiss, 3),  "avg_latency_ms": round(lat_faiss, 1)},
        "hybrid": {"p_at_5": round(avg_hybrid, 3), "avg_latency_ms": round(lat_hybrid, 1)},
        "hybrid_vs_bm25_pct":  round(hybrid_vs_bm25, 1),
        "hybrid_vs_faiss_pct": round(hybrid_vs_faiss, 1),
    }
    out_path = RESULTS_DIR / "benchmark.json"
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n✓ Benchmark saved → {out_path}")
    return benchmark


if __name__ == "__main__":
    run_benchmark()
