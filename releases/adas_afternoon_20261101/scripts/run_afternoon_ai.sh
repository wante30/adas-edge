#!/bin/sh
set -eu

ROOT="${ADAS_ROOT:-/home/cat/adas_afternoon_20261101/adas_hazard_scene_20261031}"
MODEL="${ADAS_MAIN_MODEL:-/home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn}"
FRAME="${ADAS_LIVE_FRAME:-/dev/shm/qt_gpu_probe_ai.mfr1}"

cd "$ROOT"

exec env \
  ADAS_EDGE_ROOT="$ROOT" \
  ADAS_DISABLE_ROAD_SCENE=1 \
  ADAS_LEGACY_HAZARD=1 \
  ADAS_AUTO_BRAKE=1 \
  /usr/bin/python3 "$ROOT/b05p_ai_bridge_launcher.py" \
  --live-frame-file "$FRAME" \
  --source PGV_LIVE \
  --layout quad \
  --display-view front \
  --road-quadrant 0 \
  --navigation-quadrant 1 \
  --main-model "$MODEL" \
  --main-hz 10 \
  --hazard-hz 5 \
  --ocr-hz 2 \
  --navigation-ocr \
  --idle-timeout 0 \
  --timing-log /tmp/afternoon_ai.jsonl \
  --summary /tmp/afternoon_ai_summary.json
