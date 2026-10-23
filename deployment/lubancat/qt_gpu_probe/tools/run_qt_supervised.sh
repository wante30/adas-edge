#!/usr/bin/env bash
# Foreground launcher that records the exact Qt exit status and signal.
# It does not change HMI arguments, services, or the running instance.
set -u

ROOT="${QT_PROBE_ROOT:-/home/cat/qt_gpu_probe_ai_20261019}"
BIN="${ROOT}/qt_gpu_probe"
LOG="${QT_EXIT_LOG:-${ROOT}/results/qt_exit.log}"

mkdir -p "$(dirname "${LOG}")"
ulimit -c unlimited 2>/dev/null || true

{
    echo "=== $(date -Is) launch ==="
    echo "commit=$(git -C /home/cat/adas-edge rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "binary=$(sha256sum "${BIN}" 2>/dev/null | cut -d' ' -f1)"
    echo "core_pattern=$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo unknown)"
    echo "cmd=$(printf '%q ' "${BIN}" "$@")"
} >>"${LOG}"

"${BIN}" "$@"
rc=$?

sig=0
if [ "${rc}" -gt 128 ]; then
    sig=$((rc - 128))
fi
sig_name="none"
if [ "${sig}" -gt 0 ]; then
    sig_name=$(kill -l "${sig}" 2>/dev/null || echo "signal-${sig}")
fi
{
    echo "exit_code=${rc}"
    echo "signal=${sig_name}"
    echo "=== $(date -Is) end ==="
} >>"${LOG}"
exit "${rc}"
