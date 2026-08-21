#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="${VLLM_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM_ROOT}/.venv/bin/python}"

MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
USE_BREAKABLE_CUDAGRAPH="${USE_BREAKABLE_CUDAGRAPH:-1}"
VLLM_EXTRA_ENGINE_ARGS="${VLLM_EXTRA_ENGINE_ARGS:-}"

RESULT_DIR="${RESULT_DIR:-${VLLM_ROOT}/bench-results-ft-no-failure}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
READY_CHECK_TIMEOUT_SEC="${READY_CHECK_TIMEOUT_SEC:-1800}"
REPETITIONS="${REPETITIONS:-3}"
WORKLOADS="${WORKLOADS:-decode prefill}"
CONFIGS="${CONFIGS:-nccl ft-nccl}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"

FT_COLLECTIVE_PYTHON="${FT_COLLECTIVE_PYTHON:-/workspace/nccl/contrib/fault_tolerant_collectives/ft_handle/python}"
FT_NCCL_MAX_COUNT="${FT_NCCL_MAX_COUNT:-4194304}"
FT_TIMEOUT_US="${FT_TIMEOUT_US:-2000000}"
FT_BARRIER_ROUNDS="${FT_BARRIER_ROUNDS:-2}"
FT_BARRIER_MODE="${FT_BARRIER_MODE:-collective}"

DECODE_INPUT_LEN="${DECODE_INPUT_LEN:-128}"
DECODE_OUTPUT_LEN="${DECODE_OUTPUT_LEN:-1024}"
DECODE_NUM_PROMPTS="${DECODE_NUM_PROMPTS:-512}"
DECODE_NUM_WARMUPS="${DECODE_NUM_WARMUPS:-64}"
DECODE_REQUEST_RATE="${DECODE_REQUEST_RATE:-inf}"
DECODE_MAX_CONCURRENCY="${DECODE_MAX_CONCURRENCY:-256}"

PREFILL_INPUT_LEN="${PREFILL_INPUT_LEN:-2048}"
PREFILL_OUTPUT_LEN="${PREFILL_OUTPUT_LEN:-128}"
PREFILL_NUM_PROMPTS="${PREFILL_NUM_PROMPTS:-512}"
PREFILL_NUM_WARMUPS="${PREFILL_NUM_WARMUPS:-64}"
PREFILL_REQUEST_RATE="${PREFILL_REQUEST_RATE:-inf}"
PREFILL_MAX_CONCURRENCY="${PREFILL_MAX_CONCURRENCY:-256}"

is_true() {
  case "${1:-}" in
    1 | true | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_environment() {
  if [[ ! -x "${VLLM_PYTHON}" ]]; then
    echo "Expected executable vLLM Python at ${VLLM_PYTHON}" >&2
    exit 1
  fi
  if ! [[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPETITIONS must be a positive integer, got '${REPETITIONS}'" >&2
    exit 1
  fi
}

configure_huggingface_auth() {
  if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
  fi
  if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE}" ]]; then
    if [[ ! -r "${HF_TOKEN_FILE}" ]]; then
      echo "HF_TOKEN_FILE is not readable: ${HF_TOKEN_FILE}" >&2
      exit 1
    fi
    IFS= read -r HF_TOKEN <"${HF_TOKEN_FILE}" || true
    if [[ -z "${HF_TOKEN}" ]]; then
      echo "HF_TOKEN_FILE is empty: ${HF_TOKEN_FILE}" >&2
      exit 1
    fi
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
  fi
}

build_engine_args() {
  ENGINE_ARGS=(
    serve "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --data-parallel-size "${DP_SIZE}"
    --data-parallel-size-local "${DP_SIZE}"
    --data-parallel-backend mp
    --enable-expert-parallel
    --all2all-backend allgather_reducescatter
    --trust-remote-code
    --dtype "${DTYPE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  )

  if is_true "${ENFORCE_EAGER}"; then
    ENGINE_ARGS+=(--enforce-eager)
  fi
  if [[ -n "${VLLM_EXTRA_ENGINE_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=(${VLLM_EXTRA_ENGINE_ARGS})
    ENGINE_ARGS+=("${extra_args[@]}")
  fi
}

run_vllm_cli() {
  cd "${VLLM_ROOT}"
  "${VLLM_PYTHON}" -m vllm.entrypoints.cli.main "$@"
}

run_workload() {
  local config="$1"
  local workload="$2"
  local repetition="$3"
  local input_len output_len num_prompts num_warmups request_rate max_concurrency

  case "${workload}" in
    decode)
      input_len="${DECODE_INPUT_LEN}"
      output_len="${DECODE_OUTPUT_LEN}"
      num_prompts="${DECODE_NUM_PROMPTS}"
      num_warmups="${DECODE_NUM_WARMUPS}"
      request_rate="${DECODE_REQUEST_RATE}"
      max_concurrency="${DECODE_MAX_CONCURRENCY}"
      ;;
    prefill)
      input_len="${PREFILL_INPUT_LEN}"
      output_len="${PREFILL_OUTPUT_LEN}"
      num_prompts="${PREFILL_NUM_PROMPTS}"
      num_warmups="${PREFILL_NUM_WARMUPS}"
      request_rate="${PREFILL_REQUEST_RATE}"
      max_concurrency="${PREFILL_MAX_CONCURRENCY}"
      ;;
    *)
      echo "Unknown workload '${workload}'; expected decode or prefill" >&2
      return 1
      ;;
  esac

  local filename="${config}-${workload}-r${repetition}-${RUN_ID}.json"
  local args=(
    bench serve
    --backend openai
    --host "${HOST}"
    --port "${PORT}"
    --model "${MODEL}"
    --dataset-name random
    --input-len "${input_len}"
    --output-len "${output_len}"
    --num-prompts "${num_prompts}"
    --num-warmups "${num_warmups}"
    --request-rate "${request_rate}"
    --max-concurrency "${max_concurrency}"
    --ignore-eos
    --ready-check-timeout-sec "${READY_CHECK_TIMEOUT_SEC}"
    --save-result
    --result-dir "${RESULT_DIR}"
    --result-filename "${filename}"
    --metadata
    "config=${config}"
    "workload=${workload}"
    "repetition=${repetition}"
    "tp=${TP_SIZE}"
    "dp=${DP_SIZE}"
    "enforce_eager=${ENFORCE_EAGER}"
    "breakable_cudagraph=${USE_BREAKABLE_CUDAGRAPH}"
    "ft_max_count=${FT_NCCL_MAX_COUNT}"
  )

  if [[ -n "${BENCH_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=(${BENCH_EXTRA_ARGS})
    args+=("${extra_args[@]}")
  fi
  run_vllm_cli "${args[@]}"
}
