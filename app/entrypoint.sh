#!/usr/bin/env bash
set -e

MODEL="llama3.2:1b"

echo "Starting Ollama server..."
ollama serve > /tmp/ollama.log 2>&1 &

echo "Waiting for Ollama to become ready..."
for i in {1..60}; do
  if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
    break
  fi
  sleep 1
done

echo "Ollama is ready. Ensuring $MODEL model is present..."
ollama list | grep -q "$MODEL" || ollama pull "$MODEL"

echo "Warming model (load into RAM)..."
ollama run "$MODEL" "OK" || true

echo "Starting player-fit worker (background)..."
uv run python -m app.jobs.run_worker &

echo "Starting FastAPI (uvicorn)..."
exec uv run uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000
