"""
All LangGraph node functions.
Each node takes AgentState, mutates it, and returns the updated state.
"""
import os
from .state import AgentState
from ..retrieval.rrf_fusion import HybridRetriever
from ..retrieval.reranker import CrossEncoderReranker
from ..llm.prompts import build_rag_prompt, GUARDRAIL_PROMPT
from ..compliance.phi_redactor import redact_phi
from ..voice.stt import transcribe
from ..voice.tts import synthesize

# Lazy singletons — initialised on first request to save startup time
_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_llm():
    """Return generate() function — Groq if key set, else Ollama."""
    if os.getenv("GROQ_API_KEY"):
        from ..llm.groq_client import generate
    else:
        from ..llm.ollama_client import generate
    return generate


# ── Nodes ───────────────────────────────────────────────────────────────────

def stt_node(state: AgentState) -> AgentState:
    """Transcribe audio_path → query. Skip if query already set (text mode)."""
    if state.get("query"):
        return state
    audio_path = state.get("audio_path", "")
    if not audio_path:
        raise ValueError("stt_node: neither 'query' nor 'audio_path' set in state.")
    state["query"] = transcribe(audio_path)
    return state


def retrieval_node(state: AgentState) -> AgentState:
    """BM25 + FAISS hybrid retrieval with RRF fusion → top 20 candidates."""
    retriever = _get_retriever()
    state["retrieved_docs"] = retriever.retrieve(state["query"])
    return state


def rerank_node(state: AgentState) -> AgentState:
    """Cross-encoder re-rank → top 5."""
    reranker = _get_reranker()
    state["reranked_docs"] = reranker.rerank(state["query"], state["retrieved_docs"])
    return state


def generation_node(state: AgentState) -> AgentState:
    """Generate answer from context using local Ollama or Groq."""
    generate = _get_llm()
    context = "\n\n".join(d["text"] for d in state["reranked_docs"])
    history = state.get("turn_history", [])

    prompt = build_rag_prompt(
        query=state["query"],
        context=context,
        history=history,
    )
    answer = generate(prompt)

    state["answer"] = answer
    state["context"] = context
    state["turn_history"] = history + [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": answer},
    ]
    return state


def guardrail_node(state: AgentState) -> AgentState:
    """
    Simple faithfulness check: ask LLM if its own answer is grounded.
    Sets guardrail_passed = True/False. Allows up to 2 generation retries.
    """
    generate = _get_llm()
    attempts = state.get("guardrail_attempts", 0)

    prompt = GUARDRAIL_PROMPT.format(
        context=state["context"],
        answer=state["answer"],
    )
    verdict = generate(prompt, temperature=0.0, max_tokens=5).strip().upper()
    passed = verdict.startswith("YES")

    state["guardrail_passed"] = passed
    state["guardrail_attempts"] = attempts + 1
    return state


def redaction_node(state: AgentState) -> AgentState:
    """Presidio PHI redaction on the generated answer."""
    state["phi_clean_answer"] = redact_phi(state["answer"])
    return state


def tts_node(state: AgentState) -> AgentState:
    """ElevenLabs TTS (with pyttsx3 fallback) → audio_path."""
    import uuid
    filename = f"{uuid.uuid4().hex[:8]}.mp3"
    state["audio_path"] = synthesize(state["phi_clean_answer"], filename)
    return state
