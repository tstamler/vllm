#!/usr/bin/env bash
set -euo pipefail

PATTERN=${1:-Worker_DP1_TP1}
PROCESS_PREFIX=${VLLM_PROCESS_NAME_PREFIX:-VLLM}
PROCESS_REGEX="^${PROCESS_PREFIX}::${PATTERN}(_EP[0-9]+)?$"
mapfile -t pids < <(pgrep -f "${PROCESS_REGEX}" || true)

if (( ${#pids[@]} != 1 )); then
  printf 'Expected exactly one process title matching %q; found %d: %s\n' \
    "${PROCESS_REGEX}" "${#pids[@]}" "${pids[*]:-none}" >&2
  exit 1
fi

pid=${pids[0]}
ps -o pid=,ppid=,command= -p "${pid}"
printf 'Sending SIGKILL to PID %s in 3 seconds...\n' "${pid}"
sleep 3
kill -KILL "${pid}"
