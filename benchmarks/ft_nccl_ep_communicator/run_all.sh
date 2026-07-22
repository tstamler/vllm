#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

if [[ -n "${FT_EP_COMM_BENCH_WORKLOADS:-}" ]]; then
  # shellcheck disable=SC2206
  WORKLOADS=($FT_EP_COMM_BENCH_WORKLOADS)
else
  WORKLOADS=(decode prefill)
fi

if [[ $# -gt 0 ]]; then
  CONFIGS=("$@")
else
  CONFIGS=(ag-rs nixl-ep ft-ag-rs ft-a2av-single ft-a2av)
fi

server_pid=""

cleanup_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    echo "Stopping server pid=$server_pid" >&2
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}

trap cleanup_server EXIT INT TERM

write_run_info() {
  local info_file="$RESULT_DIR/run-info-$RUN_ID.txt"

  {
    echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "model=$MODEL"
    echo "tp=$TP"
    echo "dp=$DP"
    echo "host=$HOST"
    echo "port=$PORT"
    echo "dtype=$DTYPE"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "all2all_backend=$ALL2ALL_BACKEND"
    if [[ -n "${HF_TOKEN:-}" ]]; then
      echo "huggingface_auth_configured=true"
    else
      echo "huggingface_auth_configured=false"
    fi
    echo "run_id=$RUN_ID"
    echo
    git -C "$VLLM_ROOT" rev-parse HEAD 2>/dev/null || true
    git -C "$VLLM_ROOT" status --short 2>/dev/null || true
    echo
    env | sort | grep -E \
      '^(CUDA|FT_|HF_|NCCL|PYTHONPATH|RESULT_DIR|VLLM|MODEL|TP|DP|PORT|HOST)=' \
      || true
  } >"$info_file"

  nvidia-smi -q >"$RESULT_DIR/nvidia-smi-q-$RUN_ID.txt" 2>/dev/null || true
  nvidia-smi topo -m >"$RESULT_DIR/nvidia-smi-topo-$RUN_ID.txt" 2>/dev/null || true
}

server_script_for_config() {
  case "$1" in
    ag-rs) echo "$SCRIPT_DIR/serve_ag_rs.sh" ;;
    nixl-ep) echo "$SCRIPT_DIR/serve_nixl_ep.sh" ;;
    ft-ag-rs | ft-nccl-ep) echo "$SCRIPT_DIR/serve_ft_nccl_ep.sh" ;;
    ft-a2av-single) echo "$SCRIPT_DIR/serve_ft_a2av_single.sh" ;;
    ft-a2av) echo "$SCRIPT_DIR/serve_ft_a2av.sh" ;;
    *)
      echo "Unknown config '$1'. Expected ag-rs, nixl-ep, ft-ag-rs, " \
        "ft-a2av-single, or ft-a2av." >&2
      return 1
      ;;
  esac
}

backend_for_config() {
  case "$1" in
    ag-rs | ft-ag-rs | ft-nccl-ep) echo "allgather_reducescatter" ;;
    nixl-ep) echo "nixl_ep" ;;
    ft-a2av-single) echo "ft_nccl_a2av" ;;
    ft-a2av) echo "ft_nccl_ep" ;;
  esac
}

bench_script_for_workload() {
  case "$1" in
    decode) echo "$SCRIPT_DIR/bench_decode.sh" ;;
    prefill) echo "$SCRIPT_DIR/bench_prefill.sh" ;;
    *)
      echo "Unknown workload '$1'. Expected decode or prefill." >&2
      return 1
      ;;
  esac
}

write_backend_summary() {
  local config="$1"
  local server_log="$2"
  local summary_file="$RESULT_DIR/backend-summary-$RUN_ID.txt"

  {
    echo "== $config =="
    grep "Using .*all2all manager" "$server_log" || true
    grep "Using FT NCCL .*all_gatherv/reduce_scatterv" "$server_log" || true
    grep "Using FT NCCL .* all-to-allv dispatch/combine" "$server_log" || true
    grep "routed FT NCCL A2AV" "$server_log" || true
    grep "FT NCCL communicator .*fallback" "$server_log" || true
    echo
  } >>"$summary_file"
}

write_run_info

for config in "${CONFIGS[@]}"; do
  server_script="$(server_script_for_config "$config")"
  server_log="$RESULT_DIR/logs/server-${config}-${RUN_ID}.log"

  echo "Starting $config server. Log: $server_log" >&2
  "$server_script" >"$server_log" 2>&1 &
  server_pid=$!

  for workload in "${WORKLOADS[@]}"; do
    bench_script="$(bench_script_for_workload "$workload")"
    bench_log="$RESULT_DIR/logs/bench-${config}-${workload}-${RUN_ID}.log"

    echo "Running $config $workload benchmark. Log: $bench_log" >&2
    CONFIG="$config" ALL2ALL_BACKEND="$(backend_for_config "$config")" \
      "$bench_script" "$config" >"$bench_log" 2>&1
  done

  write_backend_summary "$config" "$server_log"
  cleanup_server
done

trap - EXIT INT TERM
echo "Completed FT EP communicator benchmark run. Results: $RESULT_DIR" >&2
