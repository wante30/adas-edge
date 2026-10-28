#!/bin/sh
# Start the root-owned BAR1 HUD V2 bridge used by the Qt test panel.
set -eu
RUN=/home/cat/qt_gpu_probe_ai_20261019
cd "$RUN"
sudo -n pkill -f 'hud_bridge.py' 2>/dev/null || true
sudo -n nohup python3 hud_bridge.py \
  --bar1 /sys/bus/pci/devices/0000:01:00.0/resource1 --port 38262 \
  > results/hud_bridge.log 2>&1 &
sleep 0.5
cat results/hud_bridge.log
