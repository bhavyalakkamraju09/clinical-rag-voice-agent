"""LangGraph agent state definition."""
from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    query: str                      # raw user question (text)
    retrieved_docs: list[dict]      # BM25 + FAISS RRF candidates
    reranked_docs: list[dict]       # cross-encoder top-5
    context: str                    # assembled context string for LLM
    answer: str                     # LLM-generated answer (may contain PHI)
    phi_clean_answer: str           # PHI-redacted final answer
    audio_path: str                 # path to ElevenLabs/pyttsx3 output
    turn_history: list[dict]        # multi-turn: [{"role": ..., "content": ...}]
    guardrail_passed: bool          # True if hallucination check passed
    guardrail_attempts: int         # retry counter (max 2)
    session_id: str                 # for audit logging
