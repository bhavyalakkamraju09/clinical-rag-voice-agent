"""RAG prompt templates for the clinical QA agent."""

SYSTEM_PROMPT = """You are a clinical information assistant. You answer questions \
about patient records strictly based on the provided clinical context. 

Rules:
- Answer ONLY from the provided context. Do not hallucinate.
- If the context does not contain enough information, say so clearly.
- Be concise and clinically precise.
- Never invent patient names, dates, medications, or diagnoses.
- Refer to patients as "the patient" — never by name."""


def build_rag_prompt(query: str, context: str, history: list[dict]) -> str:
    history_block = ""
    if history:
        lines = []
        for h in history[-6:]:  # last 3 turns (user + assistant)
            role = h["role"].capitalize()
            lines.append(f"{role}: {h['content']}")
        history_block = "\n".join(lines) + "\n\n"

    return f"""{SYSTEM_PROMPT}

--- CLINICAL CONTEXT ---
{context}
--- END CONTEXT ---

{history_block}User: {query}
Assistant:"""


GUARDRAIL_PROMPT = """Given the following retrieved clinical context and the model's answer, \
determine whether the answer is grounded in the context or contains hallucinated information.

Context:
{context}

Answer:
{answer}

Respond with only YES (answer is grounded) or NO (answer contains hallucinations)."""
