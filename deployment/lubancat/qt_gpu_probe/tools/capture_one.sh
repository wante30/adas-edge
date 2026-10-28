#!/usr/bin/env bash
set -u
RUN=/home/cat/qt_gpu_probe_ai_20261019
OUT=${1:-results/capture.xwd}
cd "$RUN"
WID=""
for i in 1 2 3 4 5 6 7 8 9 10; do
  WID=$(DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwininfo -root -tree 2>/dev/null \
    | grep 'Qt Quick GPU Probe' | head -1 | awk '{print $1}')
  if [ -n "$WID" ]; then break; fi
  sleep 0.5
done
if [ -z "$WID" ]; then
  echo 'window not found'
  exit 1
fi
sleep 1
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwd -id "$WID" -silent > "$OUT"
echo "captured $OUT ($WID)"
