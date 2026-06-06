"""
RAGAS evaluation — Faithfulness, Context Recall, Answer Relevancy.
Run: python evaluation/run_ragas.py
Requires: golden_test_set.json + a running agent to pre-fill model answers.
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

GOLDEN_PATH   = Path("evaluation/golden_test_set.json")
RESULTS_DIR   = Path("evaluation/results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_pipeline_on_golden(golden: list[dict]) -> list[dict]:
    """Run the full RAG pipeline on each golden question to get model answers + contexts."""
    from src.retrieval.rrf_fusion import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    from src.llm.prompts import build_rag_prompt
    import os

    if os.getenv("GROQ_API_KEY"):
        from src.llm.groq_client import generate
    else:
        from src.llm.ollama_client import generate

    retriever = HybridRetriever()
    reranker  = CrossEncoderReranker()
    results   = []

    for item in golden:
        q = item["question"]
        docs = retriever.retrieve(q)
        reranked = reranker.rerank(q, docs)
        context = "\n\n".join(d["text"] for d in reranked)
        prompt = build_rag_prompt(query=q, context=context, history=[])
        answer = generate(prompt)
        results.append({
            "question":     q,
            "answer":       answer,
            "contexts":     [d["text"] for d in reranked],
            "ground_truth": item["expected_answer"],
        })
        print(f"  [✓] {q[:60]}…")
        time.sleep(0.5)  # be nice to free-tier APIs

    return results


def main():
    if not GOLDEN_PATH.exists():
        print(f"Golden test set not found at {GOLDEN_PATH}")
        print("Generate it with: python evaluation/generate_golden_set.py")
        return

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    print(f"Running pipeline on {len(golden)} golden questions…")
    records = run_pipeline_on_golden(golden)

    dataset = Dataset.from_list(records)
    print("\nRunning RAGAS evaluation…")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )
    print("\n── RAGAS Results ─────────────────────────────")
    print(result)
    print("──────────────────────────────────────────────")

    # Save
    ts = int(time.time())
    out_path = RESULTS_DIR / f"ragas_{ts}.json"
    result.to_pandas().to_json(str(out_path), orient="records", indent=2)
    print(f"\n✓ Results saved → {out_path}")


if __name__ == "__main__":
    main()
