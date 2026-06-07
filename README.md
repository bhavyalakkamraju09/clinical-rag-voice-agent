# 🏥 Clinical RAG Voice Agent

## 🚀 Live Demo
**[Try it live on HuggingFace Spaces →](https://huggingface.co/spaces/bhavyalakkamraju09/clinical-rag-voice-agent)**

**HIPAA-Compliant Multi-Turn Voice QA over Clinical Documents**

> Ask questions about clinical notes by voice. Get spoken, PHI-redacted answers — powered by a LangGraph multi-agent pipeline with hybrid retrieval over 10,000 synthetic clinical notes.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)](https://langchain-ai.github.io/langgraph/)
[![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-orange)](https://huggingface.co/spaces/bhavyalakkamraju09/clinical-rag-voice-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📐 Architecture

```
Audio Input (Whisper STT)
        │
        ▼
   Text Query
        │
        ▼
┌─────────────────────────────────────────────────┐
│              LangGraph Agent                     │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ BM25     │  │  FAISS   │  │  RRF Fusion   │  │
│  │ Retrieval│──│  Dense   │──│  (k=60)       │  │
│  └──────────┘  └──────────┘  └───────┬───────┘  │
│                                       │          │
│                              ┌────────▼────────┐ │
│                              │ Cross-Encoder   │ │
│                              │ Re-Ranker top-5 │ │
│                              └────────┬────────┘ │
│                                       │          │
│                              ┌────────▼────────┐ │
│                              │ Llama 3.1 8B    │ │
│                              │ (Groq / Ollama) │ │
│                              └────────┬────────┘ │
│                                       │          │
│                              ┌────────▼────────┐ │
│                              │ Guardrail Node  │ │
│                              │ (hallucination) │ │
│                              └────────┬────────┘ │
│                                       │          │
│                              ┌────────▼────────┐ │
│                              │ Presidio PHI    │ │
│                              │ Redaction       │ │
│                              └────────┬────────┘ │
└───────────────────────────────────────┼──────────┘
                                        │
                               ┌────────▼────────┐
                               │ ElevenLabs TTS  │
                               │ (pyttsx3 fback) │
                               └────────┬────────┘
                                        │
                                  Audio Response
```

---

## 📊 Dataset & Scale

| Metric | Value |
|--------|-------|
| Synthetic clinical notes | **10,000** |
| Indexed chunks | **28,821** |
| Unique diagnoses | **25** |
| Note types | **5** (follow-up, new patient, procedure, discharge, urgent) |
| Unique medications | **40** |
| Lab panel types | **6** |
| Imaging report types | **10** |

---

## 🔬 Evaluation Results

| Metric | Score | Target |
|--------|-------|--------|
| LLM-as-judge Faithfulness | 0.60 | >0.85 |
| Answer Relevancy | 0.66 | >0.80 |
| Context Recall | 0.28 | >0.75 |
| **End-to-end latency (Groq)** | **6.1s ✅** | <8s |
| Avg latency (warm model) | **3.7s ✅** | <6s |

> Evaluation uses a custom LLM-as-judge framework (no RAGAS dependency) over 50 diverse clinical QA pairs spanning medications, vital signs, lab results, imaging, and treatment plans. Improvement roadmap: LoRA fine-tuning on clinical domain + query expansion.

---

## 🔒 HIPAA Compliance Features

| Feature | Implementation |
|---------|---------------|
| PHI De-identification | Microsoft Presidio — all 18 Safe Harbor identifiers |
| Audit Trail | SHA-256 hashed query/response pairs in SQLite |
| Local Inference Option | Ollama (no PHI leaves device) |
| Guardrail | Lexical grounding filter on every response |
| TTS Redaction | PHI redacted *before* speech synthesis |

---

## 🚀 Quick Start (Mac Apple Silicon)

### 1. Clone + Environment

```bash
git clone https://github.com/bhavyalakkamraju09/clinical-rag-voice-agent
cd clinical-rag-voice-agent

conda create -n clinical-rag python=3.11
conda activate clinical-rag

pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Install Ollama LLM (free, local, HIPAA-safe)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama serve
```

### 3. Environment Variables

```bash
cp .env.example .env
# Add your free API keys:
#   GROQ_API_KEY       → groq.com (free, 14,400 req/day)
#   ELEVENLABS_API_KEY → elevenlabs.io (10K chars/month free)
```

### 4. Generate Data + Build Indexes

```bash
make data     # generates 10,000 synthetic clinical notes
make index    # builds FAISS + BM25 indexes (~8 min)
```

### 5. Launch

```bash
# Terminal 1 — FastAPI backend
make run-api

# Terminal 2 — Streamlit UI
make run-ui
# → Open http://localhost:8501
```

---

## 📂 Project Structure

```
clinical-rag-voice-agent/
├── data/
│   ├── synthetic/generate_synthetic_notes.py   # 10,000 synthetic notes (Faker)
│   ├── raw/                                     # generated notes (gitignored)
│   └── processed/                               # FAISS index + BM25 pkl
├── src/
│   ├── ingestion/    chunker · embedder · indexer
│   ├── retrieval/    bm25 · faiss · rrf_fusion · reranker
│   ├── agent/        state · nodes · graph · memory
│   ├── llm/          ollama_client · groq_client · prompts
│   ├── voice/        stt (Whisper) · tts (ElevenLabs + pyttsx3)
│   ├── compliance/   phi_redactor · guardrails · audit_logger
│   └── api/          FastAPI routes
├── app/
│   └── streamlit_app.py
├── fine_tuning/
│   ├── prepare_dataset.py
│   ├── train_lora_colab.py     # run on Colab T4 (free)
│   └── merge_adapter.py
├── evaluation/
│   ├── create_golden_set.py    # 50 diverse clinical QA pairs
│   ├── run_evaluation.py       # custom LLM-as-judge framework
│   └── results/
├── docker-compose.yml
└── Makefile
```

---

## 💡 Why Each Design Choice?

**RRF over weighted fusion** — Parameter-free, robust to score distribution differences between BM25 sparse scores and FAISS cosine similarities. No alpha to tune.

**LangGraph over vanilla LangChain** — Typed state machine with conditional edges makes the guardrail retry loop, multi-turn memory, and future human-in-the-loop first-class citizens.

**Two-stage retrieval (RRF → cross-encoder)** — Bi-encoder FAISS is fast but coarse. Cross-encoder sees (query, passage) jointly for precise relevance. Pre-filtering to top-10 avoids O(N) cross-encoder cost.

**Presidio over regex** — NER-based PHI detection covers partial matches, contextual PHI, and all 18 Safe Harbor types that regex misses.

**Groq for inference** — llama-3.1-8b-instant at 14,400 free req/day drops latency from ~60s (CPU Ollama) to 6s. Ollama remains available for fully local/offline HIPAA deployments.

---

## 🎤 Voice Pipeline

```python
from src.agent.graph import get_graph

agent = get_graph()
result = agent.invoke({
    "audio_path": "patient_question.wav",   # Whisper STT
    "turn_history": [],
    "guardrail_passed": False,
    "guardrail_attempts": 0,
})
print(result["phi_clean_answer"])           # PHI-redacted answer
# result["audio_path"] → ElevenLabs .mp3
```

---

## 📊 LoRA Fine-Tuning (Colab T4)

```bash
# 1. Prepare dataset locally
make finetune-prep
# → fine_tuning/clinical_qa.jsonl

# 2. Upload to Colab T4 and run:
#    fine_tuning/train_lora_colab.py
#    ~2hrs, r=16 LoRA on q_proj + v_proj of Llama 3.1 8B

# 3. Merge and import into Ollama
python fine_tuning/merge_adapter.py
```

---

## 🐳 Docker Deployment

```bash
make docker-up
# API     → http://localhost:8000
# UI      → http://localhost:8501
# MLflow  → http://localhost:5000
```

---

## 🔑 Free Tier Reference

| Service | Free Allowance |
|---------|---------------|
| Ollama (local) | Unlimited |
| Groq API | 14,400 req/day on llama-3.1-8b-instant |
| ElevenLabs | 10,000 chars/month |
| HuggingFace Spaces | Free CPU hosting |
| Colab T4 | ~4 hrs/day for LoRA fine-tuning |
| FAISS / Presidio / MLflow | Unlimited (local) |

---

## 📝 Resume Bullets

- Built end-to-end HIPAA-compliant clinical voice QA system over 10,000 synthetic notes and 28,821 indexed chunks using LangGraph orchestration, BM25+FAISS hybrid retrieval with RRF fusion, and cross-encoder re-ranking; deployed live on HuggingFace Spaces
- Achieved 6.1s end-to-end latency via Groq inference (llama-3.1-8b-instant) with sub-40ms hybrid retrieval across 28K clinical chunks spanning 25 diagnoses, 5 note types, and 40 medications
- Implemented HIPAA-compliant PHI de-identification using Microsoft Presidio covering all 18 Safe Harbor identifiers with SHA-256-hashed audit logging; PHI redacted before TTS synthesis
- Built custom LLM-as-judge evaluation framework measuring faithfulness, answer relevancy, and context recall over 50 diverse clinical QA pairs; improvement roadmap includes LoRA fine-tuning and query expansion
- Integrated Whisper STT + ElevenLabs TTS into LangGraph agent node graph enabling fully voice-driven clinical QA; containerized full stack with Docker Compose

---

*Built by Bhavya Lakkamraju · MS Computer Science, Lawrence Technological University · 2026*  
*GitHub: [bhavyalakkamraju09](https://github.com/bhavyalakkamraju09) · LinkedIn: [bhavya-varma](https://linkedin.com/in/bhavya-varma)*
