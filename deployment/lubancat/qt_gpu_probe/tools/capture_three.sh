#!/usr/bin/env bash
# Assumes the HUD is already running with --capture-cycle (5 s per mode).
RUN=/home/cat/qt_gpu_probe_ai_20261019
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
echo "window=$WID"
sleep 1
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwd -id "$WID" -silent > results/final_single.xwd
echo 'single'
sleep 5
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwd -id "$WID" -silent > results/final_quad.xwd
echo 'quad'
sleep 5
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority xwd -id "$WID" -silent > results/final_spatial.xwd
echo 'spatial'
