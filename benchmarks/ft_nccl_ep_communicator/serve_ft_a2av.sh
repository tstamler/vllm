#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_vllm_python
ALL2ALL_BACKEND=ft_nccl_ep
# The initial PoC uses dynamic PyTorch routing and host-visible receive counts.
# Keep eager as the default until these steps are replaced by graph-safe kernels.
build_engine_args

if [[ -d "$FT_COLLECTIVE_PYTHON" ]]; then
  export PYTHONPATH="$FT_COLLECTIVE_PYTHON:${PYTHONPATH:-}"
else
  echo "FT_COLLECTIVE_PYTHON does not exist: $FT_COLLECTIVE_PYTHON" >&2
  exit 1
fi

export VLLM_USE_FT_NCCL_COMMUNICATOR=0
export VLLM_USE_FT_NCCL_EP=0
export VLLM_USE_FT_NCCL_TP=0
export VLLM_DISABLE_PYNCCL=0
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"

echo "Starting routed FT NCCL A2AV EP server: model=$MODEL tp=$TP dp=$DP" >&2
echo "FT_COLLECTIVE_PYTHON=$FT_COLLECTIVE_PYTHON" >&2
cd "$VLLM_ROOT"
exec "$VLLM_PYTHON" -m vllm.entrypoints.cli.main "${ENGINE_ARGS[@]}"
