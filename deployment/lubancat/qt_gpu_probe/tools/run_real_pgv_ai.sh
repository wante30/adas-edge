#!/usr/bin/env bash
# Real-PCIe Qt + AI integration run. Assumes pango_video_capture is already
# publishing /dev/shm/pango-video/latest.pgv.
set -euo pipefail
RUN=/home/cat/qt_gpu_probe_ai_20261019
SHM=/dev/shm/pango-video
AI_FRAME=/dev/shm/qt_gpu_probe_ai.mfr1
EDGE=/home/cat/competition_cockpit_port_20261018
MODELS=/home/cat/adas-edge-rect
DURATION=${1:-15}
cd "$RUN"

pkill -f 'ai_bridge.py --live-frame-file /dev/shm/qt_gpu_probe_ai.mfr1' 2>/dev/null || true
pkill -f './qt_gpu_probe --fps' 2>/dev/null || true
sleep 1
rm -f "$AI_FRAME"

ADAS_EDGE_ROOT="$EDGE" /usr/bin/python3 "$RUN/ai_bridge.py" \
  --live-frame-file "$AI_FRAME" --source PGV_LIVE \
  --layout quad --display-view front --road-quadrant 0 --navigation-quadrant 1 \
  --main-model "$MODELS/rect_models/yolo11n_rk3566_int8_split_512x288.rknn" \
  --specialist-model "$MODELS/tl_check/tl_specialist_512x192_hybrid2.rknn" \
  --main-hz 10 --specialist-hz 1 --ocr-hz 0.1 --idle-timeout 90 \
  --navigation-ocr \
  --summary "$RUN/results/ai_real_pgv_summary.json" \
  > "$RUN/results/ai_real_pgv.log" 2>&1 &
AI_PID=$!
echo "ai_pid=$AI_PID"
for i in $(seq 1 60); do
  if grep -q 'READY' "$RUN/results/ai_real_pgv.log" 2>/dev/null; then break; fi
  if ! kill -0 "$AI_PID" 2>/dev/null; then echo 'AI sidecar exited'; tail -80 "$RUN/results/ai_real_pgv.log"; exit 1; fi
  sleep 0.5
done
if ! grep -q 'READY' "$RUN/results/ai_real_pgv.log"; then
  echo 'AI sidecar did not produce READY'; tail -80 "$RUN/results/ai_real_pgv.log"; kill "$AI_PID" 2>/dev/null || true; exit 1
fi
echo 'AI READY'

DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl QSG_INFO=1 \
  ./qt_gpu_probe --fps 60 --duration "$DURATION" \
  --pgv-file "$SHM/latest.pgv" --source-label pgv-live \
  --pgv-layout quad --road-quadrant 0 \
  --ai-frame-file "$AI_FRAME" --ai-hz 10 \
  --output-json results/qt_real_pgv.json \
  > results/qt_real_pgv.log 2>&1 &
APP_PID=$!
echo "app_pid=$APP_PID"
sleep 6
WID=$(DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwininfo -root -tree 2>/dev/null \
  | grep 'Qt Quick GPU Probe' | head -1 | awk '{print $1}')
echo "window=$WID"
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwd -id "$WID" -silent > results/qt_real_pgv.xwd
wait $APP_PID
kill "$AI_PID" 2>/dev/null || true
wait "$AI_PID" 2>/dev/null || true
echo '--- capture status ---'
cat "$SHM/status.json" 2>/dev/null || true
echo
echo '--- qt report ---'
cat results/qt_real_pgv.json
echo
echo '--- ai summary ---'
cat results/ai_real_pgv_summary.json 2>/dev/null || true
