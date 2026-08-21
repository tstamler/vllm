#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_environment
configure_huggingface_auth
mkdir -p "${RESULT_DIR}/logs"

# Permit command-line config selection while keeping CONFIGS available for CI.
if [[ $# -gt 0 ]]; then
  configs=("$@")
else
  # shellcheck disable=SC2206
  configs=(${CONFIGS})
fi
# shellcheck disable=SC2206
workloads=(${WORKLOADS})

server_pid=""
cleanup_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "${server_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${server_pid}" 2>/dev/null; then
      kill -KILL "${server_pid}" 2>/dev/null || true
    fi
    wait "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
}
trap cleanup_server EXIT INT TERM

write_run_info() {
  local info_file="${RESULT_DIR}/run-info-${RUN_ID}.txt"
  {
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "run_id=${RUN_ID}"
    echo "model=${MODEL}"
    echo "tp=${TP_SIZE}"
    echo "dp=${DP_SIZE}"
    echo "dtype=${DTYPE}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    echo "enforce_eager=${ENFORCE_EAGER}"
    echo "repetitions=${REPETITIONS}"
    echo "configs=${configs[*]}"
    echo "workloads=${workloads[*]}"
    echo "ft_nccl_max_count=${FT_NCCL_MAX_COUNT}"
    echo "ft_timeout_us=${FT_TIMEOUT_US}"
    echo "ft_barrier_rounds=${FT_BARRIER_ROUNDS}"
    echo "ft_barrier_mode=${FT_BARRIER_MODE}"
    echo "huggingface_auth_configured=$([[ -n "${HF_TOKEN:-}" ]] && echo true || echo false)"
    echo
    git -C "${VLLM_ROOT}" rev-parse HEAD 2>/dev/null || true
    git -C "${VLLM_ROOT}" status --short 2>/dev/null || true
  } >"${info_file}"

  nvidia-smi -q >"${RESULT_DIR}/nvidia-smi-q-${RUN_ID}.txt" 2>/dev/null || true
  nvidia-smi topo -m >"${RESULT_DIR}/nvidia-smi-topo-${RUN_ID}.txt" 2>/dev/null || true
}

wait_for_server() {
  local deadline=$((SECONDS + READY_CHECK_TIMEOUT_SEC))
  while ((SECONDS < deadline)); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Server exited before becoming ready" >&2
      return 1
    fi
    if curl --fail --silent "http://${HOST}:${PORT}/health" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "Server did not become ready within ${READY_CHECK_TIMEOUT_SEC}s" >&2
  return 1
}

write_run_info

for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  for config in "${configs[@]}"; do
    server_log="${RESULT_DIR}/logs/server-${config}-r${repetition}-${RUN_ID}.log"
    echo "Starting ${config}, repetition ${repetition}; log=${server_log}" >&2
    "${SCRIPT_DIR}/serve.sh" "${config}" >"${server_log}" 2>&1 &
    server_pid=$!
    if ! wait_for_server; then
      tail -100 "${server_log}" >&2 || true
      exit 1
    fi

    for workload in "${workloads[@]}"; do
      bench_log="${RESULT_DIR}/logs/bench-${config}-${workload}-r${repetition}-${RUN_ID}.log"
      echo "Running ${config} ${workload}, repetition ${repetition}" >&2
      run_workload "${config}" "${workload}" "${repetition}" \
        >"${bench_log}" 2>&1
    done
    cleanup_server
  done
done

"${VLLM_PYTHON}" "${SCRIPT_DIR}/summarize.py" \
  --result-dir "${RESULT_DIR}" \
  --run-id "${RUN_ID}" \
  --output "${RESULT_DIR}/summary-${RUN_ID}.csv"

trap - EXIT INT TERM
echo "Completed no-failure benchmarks: ${RESULT_DIR}" >&2
