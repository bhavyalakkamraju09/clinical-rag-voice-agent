FROM python:3.11-slim

WORKDIR /app

# System deps for audio (pyttsx3, whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg libespeak1 espeak \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_lg

COPY . .

EXPOSE 8000 8501
