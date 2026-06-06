"""
Groq API client — llama-3.1-8b-instant (free tier, fast for demos).
Falls back to Ollama if GROQ_API_KEY not set.
"""
import os
from groq import Groq

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("GROQ_API_KEY not set. Use Ollama instead.")
        _client = Groq(api_key=key)
    return _client


def generate(prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
