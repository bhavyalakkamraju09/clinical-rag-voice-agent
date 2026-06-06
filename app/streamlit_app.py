"""
Clinical RAG Voice Agent — Redesigned UI
Run: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ClinicalRAG",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #0a0f1a !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e2e8f0 !important;
}

/* ── Hide Streamlit chrome ──────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 2rem 2.5rem 2rem !important; max-width: 1200px !important; }

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid #1e2d47 !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] .sidebar-brand {
    color: #e2e8f0 !important;
}

/* ── Header area ────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 8px;
}
.app-logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.app-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-subtitle {
    font-size: 0.8rem;
    color: #475569;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Status pill ────────────────────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    margin-bottom: 24px;
}
.status-online {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #10b981;
}
.status-offline {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #ef4444;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Chat container ─────────────────────────────────────────── */
.chat-wrap {
    min-height: 120px;
    margin-bottom: 24px;
}

/* ── Message bubbles ────────────────────────────────────────── */
.msg-row {
    display: flex;
    margin-bottom: 20px;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user  { justify-content: flex-end; }
.msg-row.agent { justify-content: flex-start; }

.msg-avatar {
    width: 32px; height: 32px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar-user  { background: linear-gradient(135deg, #0ea5e9, #6366f1); margin-left: 10px; }
.avatar-agent { background: #1e2d47; margin-right: 10px; }

.msg-bubble {
    max-width: 68%;
    padding: 13px 16px;
    border-radius: 16px;
    font-size: 0.9rem;
    line-height: 1.6;
    font-weight: 400;
}
.bubble-user {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: #fff;
    border-radius: 16px 16px 4px 16px;
}
.bubble-agent {
    background: #131d2e;
    color: #cbd5e1;
    border: 1px solid #1e2d47;
    border-radius: 16px 16px 16px 4px;
}

.msg-meta {
    font-size: 0.7rem;
    color: #334155;
    margin-top: 5px;
    padding: 0 4px;
    text-align: right;
}
.msg-row.agent .msg-meta { text-align: left; }

/* ── Input row ──────────────────────────────────────────────── */
.stTextInput input {
    background: #131d2e !important;
    border: 1px solid #1e2d47 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.12) !important;
}
.stTextInput input::placeholder { color: #334155 !important; }

/* ── Buttons ────────────────────────────────────────────────── */
.stButton button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    transition: opacity 0.2s, transform 0.1s !important;
    cursor: pointer !important;
}
.stButton button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton button:active { transform: translateY(0) !important; }

/* ── Example chips ──────────────────────────────────────────── */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
.chip {
    background: #131d2e;
    border: 1px solid #1e2d47;
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: #64748b;
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'DM Sans', sans-serif;
}
.chip:hover { border-color: #0ea5e9; color: #0ea5e9; background: rgba(14,165,233,0.06); }

/* ── Metrics row ────────────────────────────────────────────── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 24px;
}
.metric-card {
    background: #131d2e;
    border: 1px solid #1e2d47;
    border-radius: 14px;
    padding: 16px 20px;
}
.metric-label {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #f1f5f9;
    font-family: 'DM Mono', monospace;
    letter-spacing: -0.02em;
}
.metric-value.green { color: #10b981; }
.metric-value.blue  { color: #0ea5e9; }

/* ── Divider ─────────────────────────────────────────────────── */
.slim-divider {
    border: none;
    border-top: 1px solid #1a2540;
    margin: 20px 0;
}

/* ── Toggle ──────────────────────────────────────────────────── */
.stToggle label { color: #64748b !important; font-size: 0.82rem !important; }

/* ── Sidebar sections ─────────────────────────────────────────── */
.sidebar-section-title {
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #2d3f5a !important;
    font-weight: 600 !important;
    margin: 20px 0 8px !important;
}
.pipeline-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border-radius: 8px;
    margin-bottom: 3px;
    background: #111827;
    border: 1px solid #1a2540;
}
.pipeline-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #10b981;
    flex-shrink: 0;
}
.pipeline-name {
    font-size: 0.78rem !important;
    color: #64748b !important;
    font-weight: 400 !important;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2d47; border-radius: 4px; }

/* ── Audio player ────────────────────────────────────────────── */
audio {
    width: 100%;
    height: 32px;
    margin-top: 8px;
    filter: invert(1) hue-rotate(180deg) brightness(0.7);
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px;">
        <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0; letter-spacing:-0.01em;">⚕ ClinicalRAG</div>
        <div style="font-size:0.7rem; color:#2d3f5a; text-transform:uppercase; letter-spacing:0.08em; margin-top:2px;">Voice Agent v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Session</div>', unsafe_allow_html=True)
    session_id = st.text_input("", value="demo-session-001", label_visibility="collapsed")
    tts_enabled = st.toggle("Voice responses", value=True)

    st.markdown('<div class="sidebar-section-title">Pipeline</div>', unsafe_allow_html=True)
    pipeline_steps = [
        ("🎤", "Whisper STT"),
        ("🔍", "BM25 + FAISS"),
        ("⚡", "RRF Fusion"),
        ("🎯", "Cross-Encoder"),
        ("🤖", "Llama 3.1 8B"),
        ("🛡️", "Presidio PHI"),
        ("🔊", "ElevenLabs TTS"),
        ("📋", "Audit Log"),
    ]
    for icon, name in pipeline_steps:
        st.markdown(f"""
        <div class="pipeline-item">
            <div class="pipeline-dot"></div>
            <span class="pipeline-name">{icon} {name}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Actions</div>', unsafe_allow_html=True)
    if st.button("Clear Conversation", use_container_width=True):
        try:
            requests.delete(f"{API_BASE}/session/{session_id}", timeout=5)
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.metrics = []
        st.rerun()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-logo">⚕</div>
    <div>
        <div class="app-title">Clinical RAG Voice Agent</div>
        <div class="app-subtitle">HIPAA-Compliant · Multi-Turn · Voice-Enabled</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Backend status
try:
    h = requests.get(f"{API_BASE}/health", timeout=3)
    online = h.status_code == 200
except Exception:
    online = False

if online:
    st.markdown('<div class="status-pill status-online"><div class="status-dot"></div>Backend connected</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-pill status-offline"><div class="status-dot"></div>Backend offline — run: make run-api</div>', unsafe_allow_html=True)

st.markdown('<hr class="slim-divider">', unsafe_allow_html=True)

# ── Chat history ─────────────────────────────────────────────────────────────
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 40px 0; color: #1e2d47;">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">⚕</div>
        <div style="font-size: 0.85rem; color: #2d3f5a;">Ask a clinical question to get started</div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div>
                <div class="msg-bubble bubble-user">{msg["content"]}</div>
            </div>
            <div class="msg-avatar avatar-user">👤</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-row agent">
            <div class="msg-avatar avatar-agent">⚕</div>
            <div>
                <div class="msg-bubble bubble-agent">{msg["content"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if msg.get("audio_path"):
            fname = Path(msg["audio_path"]).name
            try:
                audio_resp = requests.get(f"{API_BASE}/audio/{fname}", timeout=10)
                if audio_resp.status_code == 200:
                    st.audio(audio_resp.content, format="audio/mp3")
            except Exception:
                pass

st.markdown('</div>', unsafe_allow_html=True)

# ── Example chips ─────────────────────────────────────────────────────────────
examples = [
    "What conditions are most common?",
    "List patients on Metformin",
    "Vital signs for COPD patients?",
    "Which patients have CHF?",
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, key=f"chip_{ex[:15]}", use_container_width=True):
        st.session_state.pending_query = ex
        st.rerun()

st.markdown('<hr class="slim-divider">', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([6, 1])
with col_input:
    query = st.text_input(
        "",
        placeholder="Ask about medications, diagnoses, vitals, conditions…",
        key="query_input",
        label_visibility="collapsed",
    )
with col_btn:
    send = st.button("Send →", use_container_width=True, type="primary")

# Handle pending query from chip click
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    send = True

# ── Handle send ───────────────────────────────────────────────────────────────
if send and query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Retrieving · Re-ranking · Generating…"):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{API_BASE}/query/text",
                json={"query": query, "session_id": session_id, "tts_enabled": tts_enabled},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            latency = data.get("latency_ms", (time.time() - t0) * 1000)
            st.session_state.metrics.append(latency)
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "audio_path": data.get("audio_path"),
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error: {e}",
            })
    st.rerun()

# ── Metrics ───────────────────────────────────────────────────────────────────
if st.session_state.metrics:
    avg = sum(st.session_state.metrics) / len(st.session_state.metrics)
    last = st.session_state.metrics[-1]
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-label">Avg Latency</div>
            <div class="metric-value blue">{avg/1000:.1f}s</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Queries</div>
            <div class="metric-value">{len(st.session_state.metrics)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">PHI Redaction</div>
            <div class="metric-value green">Active</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
