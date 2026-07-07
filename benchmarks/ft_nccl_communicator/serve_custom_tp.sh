#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_vllm_python
build_engine_args

export VLLM_USE_FT_NCCL_COMMUNICATOR=0
export VLLM_USE_FT_NCCL_TP=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_NCCL_SYMM_MEM=0
export VLLM_DISABLE_PYNCCL=0

echo "Starting custom TP server: model=$MODEL tp=$TP port=$PORT" >&2
cd "$VLLM_ROOT"
exec "$VLLM_PYTHON" -m vllm.entrypoints.cli.main "${ENGINE_ARGS[@]}"
