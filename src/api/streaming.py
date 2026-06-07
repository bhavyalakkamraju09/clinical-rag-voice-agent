"""
Streaming FastAPI endpoint — streams answer tokens as they're generated.
Add this to src/api/main.py or use as a standalone endpoint.
"""
import os
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class StreamQueryRequest(BaseModel):
    query: str
    session_id: str = "default"


def stream_groq(prompt: str):
    """Stream tokens from Groq API."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def add_streaming_endpoint(app: FastAPI):
    """
    Call this in src/api/main.py to add streaming support:
        from src.api.streaming import add_streaming_endpoint
        add_streaming_endpoint(app)
    """
    @app.post("/query/stream")
    async def query_stream(req: StreamQueryRequest):
        """
        Stream answer tokens as Server-Sent Events.
        Frontend usage:
            const resp = await fetch('/query/stream', {method:'POST', body: JSON.stringify({query, session_id})})
            const reader = resp.body.getReader()
            while(true) {
                const {done, value} = await reader.read()
                if (done) break
                updateUI(new TextDecoder().decode(value))
            }
        """
        from src.retrieval.rrf_fusion import HybridRetriever
        from src.retrieval.reranker import CrossEncoderReranker
        from src.llm.prompts import build_rag_prompt
        from src.agent.memory import get_history

        retriever = HybridRetriever()
        reranker  = CrossEncoderReranker()
        history   = get_history(req.session_id)

        docs      = retriever.retrieve(req.query)
        reranked  = reranker.rerank(req.query, docs)
        context   = "\n\n".join(d["text"] for d in reranked)
        prompt    = build_rag_prompt(req.query, context, history)

        return StreamingResponse(
            stream_groq(prompt),
            media_type="text/plain",
            headers={"X-Session-Id": req.session_id},
        )
