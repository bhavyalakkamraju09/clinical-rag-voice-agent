"""
Session-scoped multi-turn memory store.
In-process dict keyed by session_id. For production, swap with Redis.
"""
from threading import Lock

_store: dict[str, list[dict]] = {}
_lock = Lock()


def get_history(session_id: str) -> list[dict]:
    with _lock:
        return list(_store.get(session_id, []))


def save_history(session_id: str, history: list[dict]):
    with _lock:
        _store[session_id] = history[-20:]  # keep last 10 turns


def clear_history(session_id: str):
    with _lock:
        _store.pop(session_id, None)
