"""
Query expansion for improved context recall.
Rewrites the query 3 ways, retrieves for each, merges with RRF.
Addresses the low context recall (0.28) by increasing retrieval surface.
"""
import os


def expand_query(query: str) -> list[str]:
    """
    Generate 3 query variations using Groq.
    Returns original + 2 rewrites for broader retrieval coverage.
    """
    if os.getenv("GROQ_API_KEY"):
        from src.llm.groq_client import generate
    else:
        from src.llm.ollama_client import generate

    prompt = f"""You are a clinical information retrieval assistant.
Given this clinical question, generate 2 alternative phrasings that mean the same thing
but use different clinical terminology. This helps retrieve more relevant documents.

Original question: {query}

Respond with exactly 2 alternative phrasings, one per line, no numbering, no explanation."""

    try:
        response = generate(prompt, temperature=0.3, max_tokens=100)
        alternatives = [line.strip() for line in response.strip().split("\n") if line.strip()][:2]
        return [query] + alternatives
    except Exception:
        return [query]


def expanded_retrieve(query: str, retriever, top_n: int = 10) -> list[dict]:
    """
    Retrieve using multiple query variations and merge with RRF.
    Significantly improves context recall for paraphrased queries.
    """
    queries = expand_query(query)
    K = 60

    # Collect results from all query variations
    all_results: dict[int, float] = {}
    for q_idx, q in enumerate(queries):
        results = retriever.retrieve(q, top_n=top_n)
        # Weight original query higher (rank boost)
        weight = 1.0 if q_idx == 0 else 0.7
        for rank, doc in enumerate(results):
            doc_id = id(doc)  # use object id as proxy
            # Find actual chunk index
            for i, chunk in enumerate(retriever.chunks):
                if chunk["text"] == doc["text"]:
                    all_results[i] = all_results.get(i, 0) + weight / (K + rank + 1)
                    break

    # Sort by merged RRF score
    sorted_ids = sorted(all_results, key=all_results.__getitem__, reverse=True)[:top_n]
    return [
        {**retriever.chunks[i], "rrf_score": all_results[i]}
        for i in sorted_ids
        if i < len(retriever.chunks)
    ]
