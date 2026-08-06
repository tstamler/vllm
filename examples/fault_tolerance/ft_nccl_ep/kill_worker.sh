#!/usr/bin/env bash
set -euo pipefail

PATTERN=${1:-Worker_DP1_TP1}
mapfile -t pids < <(pgrep -f "${PATTERN}" || true)

if (( ${#pids[@]} != 1 )); then
  printf 'Expected exactly one process matching %q; found %d: %s\n' \
    "${PATTERN}" "${#pids[@]}" "${pids[*]:-none}" >&2
  exit 1
fi

pid=${pids[0]}
ps -o pid=,ppid=,command= -p "${pid}"
printf 'Sending SIGKILL to PID %s in 3 seconds...\n' "${pid}"
sleep 3
kill -KILL "${pid}"
