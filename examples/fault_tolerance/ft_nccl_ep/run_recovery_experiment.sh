#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-${ROOT_DIR}/.venv/bin/python}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
MODEL=${MODEL:-deepseek-ai/DeepSeek-V2-Lite}
DURATION_SECONDS=${DURATION_SECONDS:-600}
FAILURE_AT_SECONDS=${FAILURE_AT_SECONDS:-180}
CONCURRENCY=${CONCURRENCY:-64}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-120}
MAX_TOKENS=${MAX_TOKENS:-128}
STARTUP_RAMP_SECONDS=${STARTUP_RAMP_SECONDS:-10}
REQUEST_JITTER_SECONDS=${REQUEST_JITTER_SECONDS:-0.25}
LOAD_SEED=${LOAD_SEED:-0}
SERVER_SMOOTHING_SECONDS=${SERVER_SMOOTHING_SECONDS:-5}
CLIENT_SMOOTHING_SECONDS=${CLIENT_SMOOTHING_SECONDS:-15}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/ft-recovery-$(date +%Y%m%d-%H%M%S)}
WORKER_PATTERN=${WORKER_PATTERN:-Worker_DP1_TP1}

mkdir -p "${OUTPUT_DIR}"
printf 'timestamp_unix,event,detail\n' > "${OUTPUT_DIR}/events.csv"

curl --fail --silent --show-error "http://${HOST}:${PORT}/health" >/dev/null

collector_pid=""
failure_pid=""
cleanup() {
  [[ -z "${collector_pid}" ]] || kill "${collector_pid}" 2>/dev/null || true
  [[ -z "${failure_pid}" ]] || kill "${failure_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

printf 'Writing recovery experiment to %s\n' "${OUTPUT_DIR}"
printf 'Failure will be injected at +%ss into %s\n' \
  "${FAILURE_AT_SECONDS}" "${WORKER_PATTERN}"

"${PYTHON}" \
  "${ROOT_DIR}/examples/fault_tolerance/ft_nccl_ep/collect_recovery_metrics.py" \
  --url "http://${HOST}:${PORT}/metrics" \
  --output "${OUTPUT_DIR}/metrics.csv" \
  --duration "${DURATION_SECONDS}" &
collector_pid=$!

(
  sleep "${FAILURE_AT_SECONDS}"
  EVENT_LOG="${OUTPUT_DIR}/events.csv" \
    "${ROOT_DIR}/examples/fault_tolerance/ft_nccl_ep/kill_worker.sh" \
    "${WORKER_PATTERN}"
) &
failure_pid=$!

"${PYTHON}" \
  "${ROOT_DIR}/examples/fault_tolerance/ft_nccl_ep/run_recovery_load.py" \
  --url "http://${HOST}:${PORT}/v1/completions" \
  --model "${MODEL}" \
  --output "${OUTPUT_DIR}/client.jsonl" \
  --duration "${DURATION_SECONDS}" \
  --concurrency "${CONCURRENCY}" \
  --request-timeout "${REQUEST_TIMEOUT_SECONDS}" \
  --max-tokens "${MAX_TOKENS}" \
  --startup-ramp "${STARTUP_RAMP_SECONDS}" \
  --request-jitter "${REQUEST_JITTER_SECONDS}" \
  --seed "${LOAD_SEED}"

wait "${collector_pid}"
collector_pid=""
wait "${failure_pid}"
failure_pid=""

"${PYTHON}" \
  "${ROOT_DIR}/examples/fault_tolerance/ft_nccl_ep/plot_recovery_throughput.py" \
  --metrics "${OUTPUT_DIR}/metrics.csv" \
  --client-log "${OUTPUT_DIR}/client.jsonl" \
  --events "${OUTPUT_DIR}/events.csv" \
  --output "${OUTPUT_DIR}/throughput.png" \
  --summary "${OUTPUT_DIR}/summary.json" \
  --smoothing-seconds "${SERVER_SMOOTHING_SECONDS}" \
  --client-smoothing-seconds "${CLIENT_SMOOTHING_SECONDS}"

printf 'Graph: %s/throughput.png\n' "${OUTPUT_DIR}"
printf 'Summary: %s/summary.json\n' "${OUTPUT_DIR}"
