#!/usr/bin/env bash
set -euo pipefail

PATTERN=${1:-Worker_DP1_TP1}
STALL_SECONDS=${2:-20}
PROCESS_PREFIX=${VLLM_PROCESS_NAME_PREFIX:-VLLM}
PROCESS_REGEX="^${PROCESS_PREFIX}::${PATTERN}(_EP[0-9]+)?$"
REJOIN_TRIGGER_FILE=${VLLM_FT_REJOIN_TRIGGER_FILE:-/tmp/vllm-ft-rejoin.trigger}
REJOIN_ACK_DIR=${VLLM_FT_REJOIN_ACK_DIR:-/tmp/vllm-ft-rejoin-acks}
REJOIN_DELAY_SECONDS=${REJOIN_DELAY_SECONDS:-30}
REJOIN_ACK_TIMEOUT_SECONDS=${REJOIN_ACK_TIMEOUT_SECONDS:-60}
REJOIN_EXPECTED_ACKS=${REJOIN_EXPECTED_ACKS:-8}

mapfile -t pids < <(pgrep -f "${PROCESS_REGEX}" || true)
if (( ${#pids[@]} != 1 )); then
  printf 'Expected exactly one process title matching %q; found %d: %s\n' \
    "${PROCESS_REGEX}" "${#pids[@]}" "${pids[*]:-none}" >&2
  exit 1
fi

record_event() {
  local event=$1
  local detail=$2
  if [[ -n "${EVENT_LOG:-}" ]]; then
    printf '%s,%s,%s\n' "$(date +%s.%N)" "${event}" "${detail}" \
      >> "${EVENT_LOG}"
  fi
}

pid=${pids[0]}
ps -o pid=,ppid=,command= -p "${pid}"
kill -STOP "${pid}"
record_event worker_stalled "pid=${pid} pattern=${PATTERN}"
printf 'Stalled PID %s for %ss.\n' "${pid}" "${STALL_SECONDS}"
sleep "${STALL_SECONDS}"

kill -CONT "${pid}"
record_event worker_resumed "pid=${pid} pattern=${PATTERN}"
printf 'Resumed PID %s; allowing %ss to leave its interrupted model step.\n' \
  "${pid}" "${REJOIN_DELAY_SECONDS}"
sleep "${REJOIN_DELAY_SECONDS}"

generation=$(date +%s%N)
mkdir -p "${REJOIN_ACK_DIR}" "$(dirname "${REJOIN_TRIGGER_FILE}")"
tmp_trigger="${REJOIN_TRIGGER_FILE}.tmp.$$"
printf '%s\n' "${generation}" > "${tmp_trigger}"
mv "${tmp_trigger}" "${REJOIN_TRIGGER_FILE}"
record_event rejoin_requested "generation=${generation}"

deadline=$((SECONDS + REJOIN_ACK_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  shopt -s nullglob
  acks=("${REJOIN_ACK_DIR}/${generation}.rank-"*.ack)
  shopt -u nullglob
  if (( ${#acks[@]} >= REJOIN_EXPECTED_ACKS )); then
    record_event rejoin_complete \
      "generation=${generation} acknowledgements=${#acks[@]}"
    printf 'Rejoin generation %s completed with %d acknowledgements.\n' \
      "${generation}" "${#acks[@]}"
    exit 0
  fi
  sleep 0.2
done

record_event rejoin_timeout \
  "generation=${generation} expected=${REJOIN_EXPECTED_ACKS}"
printf 'Timed out waiting for %s acknowledgements for generation %s.\n' \
  "${REJOIN_EXPECTED_ACKS}" "${generation}" >&2
exit 1
