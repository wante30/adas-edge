#!/usr/bin/env bash
set -euo pipefail
RUN=/home/cat/qt_gpu_probe_ai_20261019
SHM=/dev/shm/pango-video
cd "$RUN"
pkill -f './qt_gpu_probe --fps' 2>/dev/null || true
sleep 1
QSG_RENDER_LOOP=basic DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl QSG_INFO=1 \
  ./qt_gpu_probe --fps 60 --duration 18 \
  --pgv-file "$SHM/latest.pgv" --source-label pgv-live \
  --output-json results/qt_pgv_basic.json \
  > results/qt_pgv_basic.log 2>&1 &
APP_PID=$!
sleep 6
WID=$(DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwininfo -root -tree 2>/dev/null \
  | grep 'Qt Quick GPU Probe' | head -1 | awk '{print $1}')
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwd -id "$WID" -silent > results/qt_pgv_basic.xwd
top -b -n 1 -H -p "$APP_PID" > results/qt_pgv_basic_top.txt 2>&1 || true
wait $APP_PID
grep -E 'render_loop|scene_graph|source_fps|successfully_uploaded|unique_source|texture_upload_p95|process_cpu' results/qt_pgv_basic.log results/qt_pgv_basic.json || true
