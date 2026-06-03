#!/usr/bin/env bash
set -e

echo "Starting player-fit worker (background)..."
uv run python -m app.jobs.run_worker &

echo "Starting FastAPI (uvicorn)..."
exec uv run uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}"
