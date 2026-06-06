"""
Whisper STT — transcribe audio file to text.
Uses openai-whisper (local, free, highly accurate on medical speech).
On Apple Silicon the model runs on MPS automatically if torch detects it.
"""
import os
import whisper

_model: whisper.Whisper | None = None
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # base is fast + accurate enough


def _get_model() -> whisper.Whisper:
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe(audio_path: str) -> str:
    """Transcribe a .wav/.mp3/.m4a file → clean text string."""
    model = _get_model()
    result = model.transcribe(audio_path, fp16=False, language="en")
    return result["text"].strip()
