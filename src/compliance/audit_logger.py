"""
HIPAA-style audit trail — logs every query/response pair to SQLite.
Query text is hashed (SHA-256); raw PHI is never stored.
"""
import hashlib
import sqlite3
import time
import os
from pathlib import Path

DB_PATH = Path(os.getenv("AUDIT_DB_PATH", "outputs/audit.db"))


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            session_id  TEXT    NOT NULL,
            query_hash  TEXT    NOT NULL,
            answer_hash TEXT    NOT NULL,
            latency_ms  REAL
        )
    """)
    conn.commit()
    return conn


def log_query(
    session_id: str,
    query: str,
    answer: str,
    latency_ms: float | None = None,
):
    """Hash query + answer and write an audit record. Never stores raw text."""
    q_hash = hashlib.sha256(query.encode()).hexdigest()
    a_hash = hashlib.sha256(answer.encode()).hexdigest()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO audit_log (timestamp, session_id, query_hash, answer_hash, latency_ms) "
        "VALUES (?, ?, ?, ?, ?)",
        (time.time(), session_id, q_hash, a_hash, latency_ms),
    )
    conn.commit()
    conn.close()


def get_audit_records(limit: int = 100) -> list[dict]:
    """Return recent audit records (no PHI — just hashes + metadata)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, timestamp, session_id, query_hash, answer_hash, latency_ms "
        "FROM audit_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    cols = ["id", "timestamp", "session_id", "query_hash", "answer_hash", "latency_ms"]
    return [dict(zip(cols, row)) for row in rows]
