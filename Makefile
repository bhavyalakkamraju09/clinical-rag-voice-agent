.PHONY: setup data index run-api run-ui eval all

## ── One-time setup ────────────────────────────────────────────────────────
setup:
	conda run -n clinical-rag pip install -r requirements.txt
	conda run -n clinical-rag python -m spacy download en_core_web_lg
	@echo "✓ Dependencies installed"
	@echo "Pull Ollama model: ollama pull llama3.1:8b"

## ── Data pipeline ─────────────────────────────────────────────────────────
data:
	python data/synthetic/generate_synthetic_notes.py

index:
	python -m src.ingestion.indexer

golden:
	python evaluation/generate_golden_set.py

## ── Run services ──────────────────────────────────────────────────────────
run-api:
	export $$(cat .env | xargs) && \
	export $(shell cat .env | xargs) && uvicorn src.api.main:app --reload --port 8000

run-ui:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui --port 5000

## ── Evaluation ────────────────────────────────────────────────────────────
eval:
	python evaluation/run_ragas.py

## ── Fine-tuning (prep locally, train on Colab) ────────────────────────────
finetune-prep:
	python fine_tuning/prepare_dataset.py
	@echo "Upload fine_tuning/clinical_qa.jsonl + fine_tuning/train_lora_colab.py to Colab"

## ── Docker ────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build -d
	@echo "Services: API → :8000 | UI → :8501 | MLflow → :5000"

docker-down:
	docker compose down

## ── Full pipeline (local, no docker) ─────────────────────────────────────
all: data index
	@echo "✓ Data + indexes built. Start services with: make run-api & make run-ui"
