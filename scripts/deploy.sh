#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=${1:-docker-compose.prod.yml}
ENV_FILE=${2:-.env}

# Charge .env si présent
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

echo "[deploy] compose file: $COMPOSE_FILE"
docker compose -f "$COMPOSE_FILE" pull || true
docker compose -f "$COMPOSE_FILE" up -d

echo "[deploy] waiting API..."
for i in {1..60}; do
  if curl -sf http://localhost:8000/ready >/dev/null; then
    echo "[deploy] API ready"; exit 0
  fi
  sleep 2
done

echo "[deploy] API not ready in time" >&2
docker compose -f "$COMPOSE_FILE" logs --no-color api || true
exit 1
