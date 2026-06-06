"""
FastAPI backend — exposes text + voice query endpoints.
Run: uvicorn src.api.main:app --reload --port 8000
"""
import os
import time
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..agent.graph import get_graph
from ..agent.memory import get_history, save_history, clear_history
from ..voice.stt import transcribe
from ..compliance.audit_logger import log_query, get_audit_records


# ── Models ──────────────────────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    tts_enabled: bool = True


class QueryResponse(BaseModel):
    answer: str
    audio_path: str | None = None
    session_id: str
    latency_ms: float


# ── App ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the graph on startup
    print("Warming up LangGraph agent…")
    _ = get_graph()
    print("✓ Agent ready")
    yield


app = FastAPI(
    title="Clinical RAG Voice Agent",
    description="HIPAA-compliant multi-turn voice QA over clinical documents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def run_query(query: str, session_id: str, tts_enabled: bool = True) -> QueryResponse:
    agent = get_graph()
    history = get_history(session_id)
    t0 = time.time()

    state = {
        "query": query,
        "turn_history": history,
        "guardrail_passed": False,
        "guardrail_attempts": 0,
        "session_id": session_id,
    }
    if not tts_enabled:
        # Skip TTS node by patching — we'll handle audio_path externally
        pass

    result = agent.invoke(state)
    latency_ms = (time.time() - t0) * 1000

    save_history(session_id, result.get("turn_history", []))
    log_query(session_id, query, result.get("phi_clean_answer", ""), latency_ms)

    return QueryResponse(
        answer=result.get("phi_clean_answer", ""),
        audio_path=result.get("audio_path"),
        session_id=session_id,
        latency_ms=round(latency_ms, 1),
    )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/query/text", response_model=QueryResponse)
async def query_text(req: TextQueryRequest):
    """Submit a text query; receive text answer + optional TTS audio path."""
    try:
        return run_query(req.query, req.session_id, req.tts_enabled)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/voice", response_model=QueryResponse)
async def query_voice(
    audio: UploadFile = File(...),
    session_id: str = "default",
):
    """Upload a .wav/.mp3 recording; Whisper transcribes + RAG pipeline runs."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        query = transcribe(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return run_query(query, session_id, tts_enabled=True)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear multi-turn conversation history for a session."""
    clear_history(session_id)
    return {"cleared": session_id}


@app.get("/audio/{filename}")
def get_audio(filename: str):
    """Serve generated TTS audio files."""
    path = Path("outputs/audio") / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(path), media_type="audio/mpeg")


@app.get("/audit")
def audit_log(limit: int = 50):
    """Return recent audit records (hashed — no raw PHI)."""
    return get_audit_records(limit)
