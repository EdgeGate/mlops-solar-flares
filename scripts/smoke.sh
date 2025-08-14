#!/usr/bin/env bash
set -euo pipefail

API_HOST=${API_HOST:-http://localhost:8000}

echo "Smoke /health" && curl -fsS "${API_HOST}/health" >/dev/null
echo "Smoke /ready"  && curl -fsS "${API_HOST}/ready"  >/dev/null
echo "Smoke /model-info" && curl -fsS "${API_HOST}/model-info" | head -c 200 || true
echo "OK"
