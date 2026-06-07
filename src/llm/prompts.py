"""RAG prompt templates for the clinical QA agent."""

SYSTEM_PROMPT = """You are a precise clinical information assistant. You answer questions \
strictly based on the provided clinical context.

Critical rules:
- Answer ONLY using information explicitly stated in the context below
- If the exact information is not in the context, say "The provided context does not contain this information"
- Never guess, infer, or use outside knowledge
- Be specific — include exact values (BP readings, medication doses, lab values) when present
- Keep answers concise and clinically accurate
- Do not refer to patients by name"""


def build_rag_prompt(query: str, context: str, history: list[dict]) -> str:
    history_block = ""
    if history:
        lines = []
        for h in history[-6:]:
            role = h["role"].capitalize()
            lines.append(f"{role}: {h['content']}")
        history_block = "\n".join(lines) + "\n\n"

    return f"""{SYSTEM_PROMPT}

--- CLINICAL CONTEXT START ---
{context}
--- CLINICAL CONTEXT END ---

{history_block}Question: {query}

Answer (based only on the context above):"""


GUARDRAIL_PROMPT = """Given the following retrieved clinical context and the model's answer, \
determine whether the answer is grounded in the context or contains hallucinated information.

Context:
{context}

Answer:
{answer}

Respond with only YES (answer is grounded) or NO (answer contains hallucinations)."""
