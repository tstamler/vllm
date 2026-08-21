#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

config="${1:?usage: serve.sh <nccl|ft-nccl>}"
require_environment
configure_huggingface_auth
build_engine_args

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
if is_true "${ENFORCE_EAGER}"; then
  export VLLM_USE_BREAKABLE_CUDAGRAPH=0
else
  export VLLM_USE_BREAKABLE_CUDAGRAPH="${USE_BREAKABLE_CUDAGRAPH}"
fi

case "${config}" in
  nccl)
    export VLLM_USE_FT_NCCL_COMMUNICATOR=0
    export VLLM_USE_FT_NCCL_EP=0
    export VLLM_FT_SURVIVE_WORKER_FAILURE=0
    ;;
  ft-nccl)
    if [[ ! -d "${FT_COLLECTIVE_PYTHON}" ]]; then
      echo "FT_COLLECTIVE_PYTHON does not exist: ${FT_COLLECTIVE_PYTHON}" >&2
      exit 1
    fi
    export PYTHONPATH="${FT_COLLECTIVE_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"
    export VLLM_USE_FT_NCCL_COMMUNICATOR=1
    export VLLM_USE_FT_NCCL_EP=1
    export VLLM_FT_SURVIVE_WORKER_FAILURE=1
    export FT_NCCL_MAX_COUNT FT_TIMEOUT_US FT_BARRIER_ROUNDS FT_BARRIER_MODE
    ;;
  *)
    echo "Unknown config '${config}'; expected nccl or ft-nccl" >&2
    exit 1
    ;;
esac

printf 'Starting %s: model=%s tp=%s dp=%s eager=%s breakable_graph=%s port=%s\n' \
  "${config}" "${MODEL}" "${TP_SIZE}" "${DP_SIZE}" \
  "${ENFORCE_EAGER}" "${VLLM_USE_BREAKABLE_CUDAGRAPH}" "${PORT}" >&2
cd "${VLLM_ROOT}"
exec "${VLLM_PYTHON}" -m vllm.entrypoints.cli.main "${ENGINE_ARGS[@]}"
