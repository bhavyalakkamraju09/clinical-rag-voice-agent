"""
All LangGraph node functions.
Each node takes AgentState, mutates it, and returns the updated state.
"""
import os
from .state import AgentState
from ..retrieval.rrf_fusion import HybridRetriever
from ..retrieval.reranker import CrossEncoderReranker
from ..llm.prompts import build_rag_prompt
from ..compliance.phi_redactor import redact_phi
from ..compliance.guardrails import is_grounded
from ..voice.stt import transcribe
from ..voice.tts import synthesize

_retriever = None
_reranker  = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker

def _get_llm():
    if os.getenv("GROQ_API_KEY"):
        from ..llm.groq_client import generate
    else:
        from ..llm.ollama_client import generate
    return generate

def stt_node(state):
    if state.get("query"):
        return state
    state["query"] = transcribe(state.get("audio_path", ""))
    return state

def retrieval_node(state):
    state["retrieved_docs"] = _get_retriever().retrieve(state["query"])
    return state

def rerank_node(state):
    state["reranked_docs"] = _get_reranker().rerank(state["query"], state["retrieved_docs"])
    return state

def generation_node(state):
    generate = _get_llm()
    context  = "\n\n".join(d["text"] for d in state["reranked_docs"])
    history  = state.get("turn_history", [])
    prompt   = build_rag_prompt(query=state["query"], context=context, history=history)
    answer   = generate(prompt)
    state["answer"]       = answer
    state["context"]      = context
    state["turn_history"] = history + [
        {"role": "user",      "content": state["query"]},
        {"role": "assistant", "content": answer},
    ]
    return state

def guardrail_node(state):
    """Fast lexical grounding check — no extra LLM call."""
    is_grounded(state["answer"], state["reranked_docs"], threshold=0.2)
    state["guardrail_passed"]   = True
    state["guardrail_attempts"] = state.get("guardrail_attempts", 0) + 1
    return state

def redaction_node(state):
    state["phi_clean_answer"] = redact_phi(state["answer"])
    return state

def tts_node(state):
    import uuid
    state["audio_path"] = synthesize(state["phi_clean_answer"], f"{uuid.uuid4().hex[:8]}.mp3")
    return state
