"""
Text-to-speech: ElevenLabs (primary) with pyttsx3 offline fallback.
ElevenLabs free tier: 10,000 chars/month — enough for ~25 demo sessions.
"""
import os
import uuid
from pathlib import Path

import requests

OUTPUT_DIR = Path("outputs/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Bella


def _elevenlabs(text: str, filename: str) -> str | None:
    if not ELEVENLABS_API_KEY:
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.75, "similarity_boost": 0.8},
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code == 200:
        path = OUTPUT_DIR / filename
        path.write_bytes(resp.content)
        return str(path)
    print(f"[TTS] ElevenLabs error {resp.status_code} — using fallback")
    return None


def _pyttsx3_fallback(text: str, filename: str) -> str:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    path = str(OUTPUT_DIR / filename.replace(".mp3", ".wav"))
    engine.save_to_file(text, path)
    engine.runAndWait()
    return path


def synthesize(text: str, filename: str | None = None) -> str:
    """Returns path to output audio file."""
    if filename is None:
        filename = f"{uuid.uuid4().hex[:8]}.mp3"
    result = _elevenlabs(text, filename)
    if result:
        return result
    return _pyttsx3_fallback(text, filename)
