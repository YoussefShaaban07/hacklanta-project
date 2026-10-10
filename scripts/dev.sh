#!/usr/bin/env bash
# Start the three local processes for development or a demo.
# Ctrl-C stops all of them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d .venv ]; then
  echo "No .venv. Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

PY="$ROOT/.venv/bin/python"
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "→ webhook receiver on :9000"
( cd backend && "$PY" -m uvicorn app.receiver:app --host 127.0.0.1 --port 9000 ) &
pids+=($!)

echo "→ backend on :8000"
# One worker, no reload: a single process must own the camera.
( cd backend && "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 ) &
pids+=($!)

echo "→ frontend on :5173"
( cd frontend && npm run dev ) &
pids+=($!)

echo
echo "Open http://localhost:5173"
wait
