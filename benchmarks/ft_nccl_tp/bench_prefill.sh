#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

if [[ $# -gt 0 ]]; then
  CONFIG="$1"
fi
CONFIG="${CONFIG:-manual}"

run_serve_benchmark \
  prefill \
  "$PREFILL_INPUT_LEN" \
  "$PREFILL_OUTPUT_LEN" \
  "$PREFILL_NUM_PROMPTS" \
  "$PREFILL_NUM_WARMUPS" \
  "$PREFILL_REQUEST_RATE" \
  "$PREFILL_MAX_CONCURRENCY"
