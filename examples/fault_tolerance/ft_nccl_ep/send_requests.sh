#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
MODEL=${MODEL:-deepseek-ai/DeepSeek-V2-Lite}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-0.2}

request=0
while true; do
  request=$((request + 1))
  code=$(curl --silent --show-error --output "/tmp/ft-request-${request}.json" \
    --write-out '%{http_code}' \
    --max-time 120 \
    --header 'Content-Type: application/json' \
    --data "{\"model\":\"${MODEL}\",\"prompt\":\"Count from one to ten. Request ${request}.\",\"max_tokens\":32,\"temperature\":0}" \
    "http://${HOST}:${PORT}/v1/completions" || true)
  printf '%s request=%d http=%s\n' "$(date -Iseconds)" "${request}" "${code:-000}"
  sleep "${INTERVAL_SECONDS}"
done
