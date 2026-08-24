#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-V2-Lite}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-2}
DP_SIZE=${DP_SIZE:-4}
FT_COLLECTIVE_PYTHON=${FT_COLLECTIVE_PYTHON:-/workspace/nccl/contrib/fault_tolerant_collectives/ft_handle/python}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}

export PYTHONPATH="${FT_COLLECTIVE_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_FT_NCCL_COMMUNICATOR=1
export VLLM_USE_FT_NCCL_EP=1
export VLLM_FT_SURVIVE_WORKER_FAILURE=1
export FT_NCCL_MAX_COUNT=${FT_NCCL_MAX_COUNT:-4194304}
# Keep the FT failure-detection controls visible at the experiment boundary.
# FT NCCL interprets the timeout in microseconds and requires at least two
# collective barrier rounds to propagate membership among surviving ranks.
export FT_TIMEOUT_US=${FT_TIMEOUT_US:-5000000}
export FT_BARRIER_ROUNDS=${FT_BARRIER_ROUNDS:-2}
# vLLM prepares the collective barrier's symmetric windows during startup,
# before a worker can be removed from the communicator.
export FT_BARRIER_MODE=${FT_BARRIER_MODE:-collective}
# Optional experiment-only control files. The worker polls the trigger between
# model steps; the FT library updates the device mask without graph recapture.
export VLLM_FT_REJOIN_TRIGGER_FILE=${VLLM_FT_REJOIN_TRIGGER_FILE:-/tmp/vllm-ft-rejoin.trigger}
export VLLM_FT_REJOIN_ACK_DIR=${VLLM_FT_REJOIN_ACK_DIR:-/tmp/vllm-ft-rejoin-acks}
export VLLM_FT_REJOIN_MAX_ATTEMPTS=${VLLM_FT_REJOIN_MAX_ATTEMPTS:-12}
export VLLM_FT_REJOIN_EXPECTED_RANKS=${VLLM_FT_REJOIN_EXPECTED_RANKS:-$((TP_SIZE * DP_SIZE))}
export VLLM_FT_REJOIN_ARRIVAL_TIMEOUT=${VLLM_FT_REJOIN_ARRIVAL_TIMEOUT:-60}
rm -f "${VLLM_FT_REJOIN_TRIGGER_FILE}"
mkdir -p "${VLLM_FT_REJOIN_ACK_DIR}"
rm -f "${VLLM_FT_REJOIN_ACK_DIR}"/*.ack \
  "${VLLM_FT_REJOIN_ACK_DIR}"/*.arrived \
  "${VLLM_FT_REJOIN_ACK_DIR}"/*.ready

engine_args=(
  "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --data-parallel-size "${DP_SIZE}"
  --data-parallel-size-local "${DP_SIZE}"
  --data-parallel-backend mp
  --enable-expert-parallel
  --all2all-backend allgather_reducescatter
  --trust-remote-code
)

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  engine_args+=(--enforce-eager)
else
  export VLLM_USE_BREAKABLE_CUDAGRAPH=1
fi

printf 'FT server: timeout_us=%s barrier_mode=%s barrier_rounds=%s eager=%s\n' \
  "${FT_TIMEOUT_US}" "${FT_BARRIER_MODE}" "${FT_BARRIER_ROUNDS}" \
  "${ENFORCE_EAGER}"

exec vllm serve "${engine_args[@]}"
