#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_vllm_python
ALL2ALL_BACKEND=nixl_ep
build_engine_args
ENGINE_ARGS+=(--enable-elastic-ep --enable-eplb)

export VLLM_USE_FT_NCCL_COMMUNICATOR=0
export VLLM_USE_FT_NCCL_EP=0
export VLLM_USE_FT_NCCL_TP=0
export VLLM_DISABLE_PYNCCL=0
export VLLM_NIXL_EP_MAX_NUM_RANKS="${VLLM_NIXL_EP_MAX_NUM_RANKS:-$DP}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"

echo "Starting NIXL EP server: model=$MODEL tp=$TP dp=$DP port=$PORT" >&2
cd "$VLLM_ROOT"
exec "$VLLM_PYTHON" -m vllm.entrypoints.cli.main "${ENGINE_ARGS[@]}"
