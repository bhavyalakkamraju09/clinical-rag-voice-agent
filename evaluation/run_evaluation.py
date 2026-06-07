"""
Custom RAG Evaluation Framework — no RAGAS dependency.
Measures Faithfulness, Answer Relevancy, and Context Recall
using LLM-as-judge via Groq.

Run: python evaluation/run_evaluation.py
"""
import sys, json, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_PATH  = Path("evaluation/golden_test_set.json")
RESULTS_DIR  = Path("evaluation/results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Scoring prompts ───────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are evaluating a RAG system answer for faithfulness.

Context provided to the system:
{context}

Generated answer:
{answer}

Is the answer faithful to the context? Does it avoid hallucinating facts not in the context?
Score from 0.0 to 1.0 where:
1.0 = completely faithful, all claims supported by context
0.5 = partially faithful, some claims not in context
0.0 = hallucinated, claims contradict or ignore context

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

RELEVANCY_PROMPT = """You are evaluating a RAG system answer for relevancy.

Question asked:
{question}

Generated answer:
{answer}

Does the answer actually address the question asked?
Score from 0.0 to 1.0 where:
1.0 = directly and completely answers the question
0.5 = partially answers the question
0.0 = does not address the question at all

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

RECALL_PROMPT = """You are evaluating a RAG system for context recall.

Question asked:
{question}

Expected answer (ground truth):
{expected}

Retrieved context:
{context}

Does the retrieved context contain the information needed to answer the question correctly?
Score from 0.0 to 1.0 where:
1.0 = context fully contains the needed information
0.5 = context partially contains the needed information  
0.0 = context does not contain the needed information

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""


def parse_score(text: str) -> float:
    """Extract float score from LLM response."""
    try:
        text = text.strip().split()[0].rstrip('.')
        score = float(text)
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def run_pipeline(question: str, generate_fn, retriever, reranker) -> dict:
    """Run full RAG pipeline on a question."""
    from src.llm.prompts import build_rag_prompt
    docs = retriever.retrieve(question)
    reranked = reranker.rerank(question, docs)
    context = "\n\n".join(d["text"] for d in reranked)
    prompt = build_rag_prompt(query=question, context=context, history=[])
    answer = generate_fn(prompt)
    return {"answer": answer, "context": context}


def main():
    if not GOLDEN_PATH.exists():
        print("Golden test set not found. Run: make golden")
        return

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    print(f"Loaded {len(golden)} golden QA pairs")

    from src.retrieval.rrf_fusion import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker

    if os.getenv("GROQ_API_KEY"):
        from src.llm.groq_client import generate
        print("Using Groq for evaluation")
    else:
        from src.llm.ollama_client import generate
        print("Using Ollama for evaluation")

    retriever = HybridRetriever()
    reranker  = CrossEncoderReranker()

    results = []
    faithfulness_scores = []
    relevancy_scores    = []
    recall_scores       = []

    print(f"\n{'─'*60}")
    print(f"{'Q#':<4} {'Faithfulness':>13} {'Relevancy':>10} {'Recall':>8}")
    print(f"{'─'*60}")

    for i, item in enumerate(golden):
        question = item["question"]
        expected = item["expected_answer"]

        try:
            # Run RAG pipeline
            t0 = time.time()
            pipeline_out = run_pipeline(question, generate, retriever, reranker)
            latency = (time.time() - t0) * 1000

            answer  = pipeline_out["answer"]
            context = pipeline_out["context"]

            # Score faithfulness
            f_score = parse_score(generate(
                FAITHFULNESS_PROMPT.format(context=context[:1500], answer=answer),
                temperature=0.0, max_tokens=5
            ))

            # Score relevancy
            r_score = parse_score(generate(
                RELEVANCY_PROMPT.format(question=question, answer=answer),
                temperature=0.0, max_tokens=5
            ))

            # Score context recall
            c_score = parse_score(generate(
                RECALL_PROMPT.format(question=question, expected=expected, context=context[:1500]),
                temperature=0.0, max_tokens=5
            ))

            faithfulness_scores.append(f_score)
            relevancy_scores.append(r_score)
            recall_scores.append(c_score)

            print(f"{i+1:<4} {f_score:>13.2f} {r_score:>10.2f} {c_score:>8.2f}  | {question[:45]}…")

            results.append({
                "question":     question,
                "expected":     expected,
                "answer":       answer,
                "context_len":  len(context),
                "faithfulness": f_score,
                "relevancy":    r_score,
                "recall":       c_score,
                "latency_ms":   round(latency, 1),
            })

            time.sleep(0.3)  # rate limit

        except Exception as e:
            print(f"{i+1:<4} [ERROR] {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(faithfulness_scores)
    avg_f = sum(faithfulness_scores) / n if n else 0
    avg_r = sum(relevancy_scores)    / n if n else 0
    avg_c = sum(recall_scores)       / n if n else 0
    avg_lat = sum(r["latency_ms"] for r in results) / len(results) if results else 0

    print(f"\n{'═'*60}")
    print(f"  EVALUATION RESULTS  ({n} questions)")
    print(f"{'═'*60}")
    print(f"  Faithfulness:      {avg_f:.3f}  (target: >0.85)")
    print(f"  Answer Relevancy:  {avg_r:.3f}  (target: >0.80)")
    print(f"  Context Recall:    {avg_c:.3f}  (target: >0.75)")
    print(f"  Avg Latency:       {avg_lat:.0f}ms")
    print(f"{'═'*60}")

    # Save results
    ts = int(time.time())
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    summary = {
        "timestamp": ts,
        "n_questions": n,
        "faithfulness": round(avg_f, 3),
        "answer_relevancy": round(avg_r, 3),
        "context_recall": round(avg_c, 3),
        "avg_latency_ms": round(avg_lat, 1),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Results saved → {out_path}")

    # Also save latest summary for README
    latest_path = RESULTS_DIR / "latest.json"
    with open(latest_path, "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "results"}, f, indent=2)
    print(f"✓ Summary saved → {latest_path}")


if __name__ == "__main__":
    main()
