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
# vLLM prepares the collective barrier's symmetric windows during startup,
# before a worker can be removed from the communicator.
export FT_BARRIER_MODE=${FT_BARRIER_MODE:-collective}

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

exec vllm serve "${engine_args[@]}"
