#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="${VLLM_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VLLM_PYTHON="${VLLM_PYTHON:-$VLLM_ROOT/.venv/bin/python}"

MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct}"
TP="${TP:-8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
VLLM_EXTRA_ENGINE_ARGS="${VLLM_EXTRA_ENGINE_ARGS:-}"

RESULT_DIR="${RESULT_DIR:-$VLLM_ROOT/bench-results-ft-tp-${TP}gpu}"
READY_CHECK_TIMEOUT_SEC="${READY_CHECK_TIMEOUT_SEC:-1800}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

FT_COLLECTIVE_PYTHON="${FT_COLLECTIVE_PYTHON:-/Users/tstamler/nccl/contrib/fault_tolerant_collectives/ft_handle/python}"
FT_NCCL_MAX_COUNT="${FT_NCCL_MAX_COUNT:-67108864}"

DECODE_INPUT_LEN="${DECODE_INPUT_LEN:-1024}"
DECODE_OUTPUT_LEN="${DECODE_OUTPUT_LEN:-512}"
DECODE_NUM_PROMPTS="${DECODE_NUM_PROMPTS:-4096}"
DECODE_NUM_WARMUPS="${DECODE_NUM_WARMUPS:-100}"
DECODE_REQUEST_RATE="${DECODE_REQUEST_RATE:-inf}"
DECODE_MAX_CONCURRENCY="${DECODE_MAX_CONCURRENCY:-256}"

PREFILL_INPUT_LEN="${PREFILL_INPUT_LEN:-4096}"
PREFILL_OUTPUT_LEN="${PREFILL_OUTPUT_LEN:-256}"
PREFILL_NUM_PROMPTS="${PREFILL_NUM_PROMPTS:-2048}"
PREFILL_NUM_WARMUPS="${PREFILL_NUM_WARMUPS:-50}"
PREFILL_REQUEST_RATE="${PREFILL_REQUEST_RATE:-inf}"
PREFILL_MAX_CONCURRENCY="${PREFILL_MAX_CONCURRENCY:-128}"

mkdir -p "$RESULT_DIR" "$RESULT_DIR/logs"

is_true() {
  case "${1:-}" in
    1 | true | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_vllm_python() {
  if [[ ! -x "$VLLM_PYTHON" ]]; then
    echo "Expected executable vLLM Python at $VLLM_PYTHON" >&2
    echo "Set VLLM_PYTHON=/path/to/.venv/bin/python if needed." >&2
    exit 1
  fi
}

build_engine_args() {
  ENGINE_ARGS=(
    serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  )

  if [[ -n "$SERVED_MODEL_NAME" ]]; then
    ENGINE_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
  fi

  if is_true "$ENFORCE_EAGER"; then
    ENGINE_ARGS+=(--enforce-eager)
  fi

  if [[ -n "$VLLM_EXTRA_ENGINE_ARGS" ]]; then
    # Intentionally split like a shell command-line override.
    # shellcheck disable=SC2206
    local extra_args=($VLLM_EXTRA_ENGINE_ARGS)
    ENGINE_ARGS+=("${extra_args[@]}")
  fi
}

run_vllm_cli() {
  require_vllm_python
  cd "$VLLM_ROOT"
  "$VLLM_PYTHON" -m vllm.entrypoints.cli.main "$@"
}

run_serve_benchmark() {
  local workload="$1"
  local input_len="$2"
  local output_len="$3"
  local num_prompts="$4"
  local num_warmups="$5"
  local request_rate="$6"
  local max_concurrency="$7"
  local config="${CONFIG:-manual}"
  local result_filename="${RESULT_FILENAME:-${config}-${workload}-${RUN_ID}.json}"

  local bench_args=(
    bench serve
    --backend openai
    --host "$HOST"
    --port "$PORT"
    --model "$MODEL"
    --dataset-name random
    --input-len "$input_len"
    --output-len "$output_len"
    --num-prompts "$num_prompts"
    --num-warmups "$num_warmups"
    --request-rate "$request_rate"
    --max-concurrency "$max_concurrency"
    --ignore-eos
    --ready-check-timeout-sec "$READY_CHECK_TIMEOUT_SEC"
    --save-result
    --result-dir "$RESULT_DIR"
    --result-filename "$result_filename"
    --metadata
    "config=$config"
    "workload=$workload"
    "tp=$TP"
    "model=$MODEL"
    "dtype=$DTYPE"
    "max_model_len=$MAX_MODEL_LEN"
    "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  )

  if [[ -n "$BENCH_EXTRA_ARGS" ]]; then
    # Intentionally split like a shell command-line override.
    # shellcheck disable=SC2206
    local extra_args=($BENCH_EXTRA_ARGS)
    bench_args+=("${extra_args[@]}")
  fi

  echo "Writing benchmark result to $RESULT_DIR/$result_filename" >&2
  run_vllm_cli "${bench_args[@]}"
}
