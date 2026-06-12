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
  decode \
  "$DECODE_INPUT_LEN" \
  "$DECODE_OUTPUT_LEN" \
  "$DECODE_NUM_PROMPTS" \
  "$DECODE_NUM_WARMUPS" \
  "$DECODE_REQUEST_RATE" \
  "$DECODE_MAX_CONCURRENCY"
