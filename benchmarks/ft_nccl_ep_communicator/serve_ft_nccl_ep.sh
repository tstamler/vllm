#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_vllm_python
build_engine_args

if [[ -d "$FT_COLLECTIVE_PYTHON" ]]; then
  export PYTHONPATH="$FT_COLLECTIVE_PYTHON:${PYTHONPATH:-}"
else
  echo "Warning: FT_COLLECTIVE_PYTHON does not exist: $FT_COLLECTIVE_PYTHON" >&2
fi

export VLLM_USE_FT_NCCL_COMMUNICATOR=0
export VLLM_USE_FT_NCCL_EP=1
export VLLM_USE_FT_NCCL_TP=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_NCCL_SYMM_MEM=0
export VLLM_DISABLE_PYNCCL=0
export FT_NCCL_MAX_COUNT
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"

echo "Starting FT NCCL Ag/Rs EP server: model=$MODEL tp=$TP dp=$DP port=$PORT" >&2
echo "FT_COLLECTIVE_PYTHON=$FT_COLLECTIVE_PYTHON" >&2
echo "FT_NCCL_MAX_COUNT=$FT_NCCL_MAX_COUNT" >&2
cd "$VLLM_ROOT"
exec "$VLLM_PYTHON" -m vllm.entrypoints.cli.main "${ENGINE_ARGS[@]}"
