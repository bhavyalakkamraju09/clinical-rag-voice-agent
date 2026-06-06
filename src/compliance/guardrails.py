"""
Hallucination guardrail utilities.
The main guardrail logic lives in agent/nodes.py (guardrail_node).
This module provides helpers for threshold-based filtering.
"""


def is_grounded(answer: str, context_chunks: list[dict], threshold: float = 0.3) -> bool:
    """
    Fast lexical grounding check (no LLM call).
    Checks what fraction of answer's content words appear in context.
    Use as a cheap pre-filter before the LLM-based guardrail.
    """
    context_text = " ".join(c["text"].lower() for c in context_chunks)
    answer_words = set(answer.lower().split())
    # Ignore stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in",
                 "and", "or", "for", "with", "on", "at", "by", "this", "that"}
    content_words = answer_words - stopwords
    if not content_words:
        return True
    overlap = sum(1 for w in content_words if w in context_text)
    return (overlap / len(content_words)) >= threshold
