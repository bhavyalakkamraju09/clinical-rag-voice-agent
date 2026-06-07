"""
MLflow experiment tracking for the Clinical RAG Voice Agent.
Logs: query, latency, retrieval scores, PHI redaction count, model used.
Run MLflow UI: mlflow ui --port 5000
"""
import os
import time
import hashlib
from pathlib import Path

import mlflow

EXPERIMENT_NAME = "clinical-rag-voice-agent"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_query(
    query: str,
    answer: str,
    retrieved_docs: list,
    reranked_docs: list,
    latency_ms: float,
    model_used: str = "groq/llama-3.1-8b-instant",
    guardrail_passed: bool = True,
    phi_entities_found: int = 0,
):
    """Log a single RAG query run to MLflow."""
    setup_mlflow()

    with mlflow.start_run():
        # ── Parameters ────────────────────────────────────────────
        mlflow.log_param("model", model_used)
        mlflow.log_param("n_retrieved", len(retrieved_docs))
        mlflow.log_param("n_reranked", len(reranked_docs))
        mlflow.log_param("query_hash", hashlib.sha256(query.encode()).hexdigest()[:12])

        # ── Metrics ───────────────────────────────────────────────
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("retrieval_candidates", len(retrieved_docs))
        mlflow.log_metric("reranked_docs", len(reranked_docs))
        mlflow.log_metric("phi_entities_found", phi_entities_found)
        mlflow.log_metric("guardrail_passed", int(guardrail_passed))
        mlflow.log_metric("answer_length_chars", len(answer))
        mlflow.log_metric("query_length_chars", len(query))

        # ── Retrieval quality metrics ─────────────────────────────
        if reranked_docs:
            avg_rrf = sum(d.get("rrf_score", 0) for d in reranked_docs) / len(reranked_docs)
            max_rrf = max(d.get("rrf_score", 0) for d in reranked_docs)
            avg_rerank = sum(d.get("rerank_score", 0) for d in reranked_docs) / len(reranked_docs)
            mlflow.log_metric("avg_rrf_score", avg_rrf)
            mlflow.log_metric("max_rrf_score", max_rrf)
            mlflow.log_metric("avg_rerank_score", avg_rerank)

        # ── Tags ──────────────────────────────────────────────────
        mlflow.set_tag("pipeline", "BM25+FAISS+RRF+CrossEncoder")
        mlflow.set_tag("phi_redacted", "yes" if phi_entities_found > 0 else "no")
        mlflow.set_tag("latency_bucket",
            "fast" if latency_ms < 4000 else
            "medium" if latency_ms < 8000 else "slow"
        )


def log_evaluation_run(
    n_questions: int,
    faithfulness: float,
    answer_relevancy: float,
    context_recall: float,
    avg_latency_ms: float,
    model_used: str = "groq/llama-3.1-8b-instant",
):
    """Log a full evaluation run to MLflow."""
    setup_mlflow()

    with mlflow.start_run(run_name=f"eval_{int(time.time())}"):
        mlflow.log_param("model", model_used)
        mlflow.log_param("n_questions", n_questions)
        mlflow.log_param("eval_framework", "custom_llm_as_judge")

        mlflow.log_metric("faithfulness", faithfulness)
        mlflow.log_metric("answer_relevancy", answer_relevancy)
        mlflow.log_metric("context_recall", context_recall)
        mlflow.log_metric("avg_latency_ms", avg_latency_ms)

        # Composite score
        composite = (faithfulness + answer_relevancy + context_recall) / 3
        mlflow.log_metric("composite_score", composite)

        mlflow.set_tag("pipeline", "BM25+FAISS+RRF+CrossEncoder")
        mlflow.set_tag("dataset", "10K_synthetic_clinical_notes")

        print(f"✓ Evaluation run logged to MLflow")
        print(f"  Composite score: {composite:.3f}")
