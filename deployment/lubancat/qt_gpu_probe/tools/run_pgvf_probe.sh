#!/usr/bin/env bash
set -euo pipefail
RUN=/home/cat/qt_gpu_probe_ai_20261019
SHM=/dev/shm/pgv_qt_fixture
cd "$RUN"
pkill -f pgvf_fixture_publisher.py 2>/dev/null || true
pkill -f './qt_gpu_probe --fps' 2>/dev/null || true
rm -rf "$SHM"
mkdir -p "$SHM"
python3 pgvf_fixture_publisher.py \
  --frame replay_ui/frame_00000000.rgb565 \
  --output "$SHM/latest.pgv" --fps 60 --duration 22 \
  > results/pgvf_fixture.log 2>&1 &
FIXTURE_PID=$!
sleep 2
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl QSG_INFO=1 \
  ./qt_gpu_probe --fps 60 --duration 12 \
  --pgv-file "$SHM/latest.pgv" --source-label pgv-fixture \
  --output-json results/pgvf_ui_probe.json \
  > results/pgvf_ui_probe.log 2>&1 &
APP_PID=$!
sleep 5
WID=$(DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwininfo -root -tree 2>/dev/null \
  | grep 'Qt Quick GPU Probe' | head -1 | awk '{print $1}')
echo "window=$WID"
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwd -id "$WID" -silent > results/pgvf_ui_probe.xwd
wait $APP_PID
kill $FIXTURE_PID 2>/dev/null || true
wait $FIXTURE_PID 2>/dev/null || true
cat results/pgvf_ui_probe.json
