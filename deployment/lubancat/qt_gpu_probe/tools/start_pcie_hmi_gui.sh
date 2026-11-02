#!/bin/sh
set -u

RUN=/home/cat/qt_gpu_probe_ai_20261019
PGV=/dev/shm/pango-video/latest.pgv
AI_FRAME=/dev/shm/qt_gpu_probe_ai.mfr1
STATE=/home/cat/.local/state/adas-pcie-hmi
LOG="$STATE/qt_gpu_probe.log"

mkdir -p "$STATE"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/cat/.Xauthority}"
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=xcb_egl
export QT_FRAME_METADATA_HZ=15

# XFCE starts this entry after X is ready. Keep the display alive across a
# transient Qt or AI restart, and leave the full-screen HMI visible.
while :; do
    if ! test -s "$PGV"; then
        sleep 1
        continue
    fi
    if ! systemctl is-active --quiet adas-ai-bridge-live.service 2>/dev/null; then
        sleep 1
        continue
    fi
    "$RUN/qt_gpu_probe" \
        --fps 60 \
        --duration 0 \
        --start-mode 0 \
        --pgv-file "$PGV" \
        --source-label pgv-live \
        --pgv-layout quad \
        --road-quadrant 0 \
        --ai-frame-file "$AI_FRAME" \
        --ai-hz 10 \
        >>"$LOG" 2>&1
    sleep 2
done
