"""
LangGraph StateGraph definition.
Includes conditional retry loop: guardrail failure → regenerate (max 2 retries).
"""
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    stt_node,
    retrieval_node,
    rerank_node,
    generation_node,
    guardrail_node,
    redaction_node,
    tts_node,
)


def _guardrail_router(state: AgentState) -> str:
    """Route to redaction if passed, back to generation if failed (max 2 retries)."""
    if state.get("guardrail_passed"):
        return "redaction"
    if state.get("guardrail_attempts", 0) >= 2:
        # Give up after 2 retries — still redact and return best answer
        return "redaction"
    return "generation"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("stt",        stt_node)
    graph.add_node("retrieval",  retrieval_node)
    graph.add_node("rerank",     rerank_node)
    graph.add_node("generation", generation_node)
    graph.add_node("guardrail",  guardrail_node)
    graph.add_node("redaction",  redaction_node)
    graph.add_node("tts",        tts_node)

    graph.set_entry_point("stt")
    graph.add_edge("stt",        "retrieval")
    graph.add_edge("retrieval",  "rerank")
    graph.add_edge("rerank",     "generation")
    graph.add_edge("generation", "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        _guardrail_router,
        {"redaction": "redaction", "generation": "generation"},
    )
    graph.add_edge("redaction",  "tts")
    graph.add_edge("tts",        END)

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
