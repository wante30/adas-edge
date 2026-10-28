#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BOARD_ROOT="/home/cat/adas-edge-rect"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
OUTPUT_DIR="$APP_ROOT/output/competition_live_$(date +%Y%m%d_%H%M%S)"

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/cat/.Xauthority}"

if [[ -n "${PGV_FILE:-}" ]]; then
  SOURCE_ARGS=(--pgv "$PGV_FILE")
  PGV_ARGS=(--pgv-quadrant "${PGV_QUADRANT:-3}" \
            --pgv-navigation-quadrant "${PGV_NAVIGATION_QUADRANT:-1}")
else
  SOURCE_ARGS=(--video "$BOARD_ROOT/videos/front_eval_intersection_20261001.mp4" --paused)
  PGV_ARGS=()
fi

if [[ "${FPGA_HUD:-0}" == "1" && -z "${PGV_FILE:-}" ]]; then
  echo "FPGA_HUD=1 requires PGV_FILE to name a live PCIe capture." >&2
  exit 2
fi

MODE_ARGS=(--mode intersection)
FPGA_ARGS=()
HAZARD_ARGS=()
if [[ "${FPGA_HUD:-0}" == "1" ]]; then
  MODE_ARGS=()
  FPGA_ARGS=(--fpga-hud)
fi
if [[ "${HAZARDS:-0}" == "1" ]]; then
  HAZARD_ARGS=(--hazards)
fi

RUNNER_ARGS=(
  "$APP_ROOT/tools/run_competition_console.py"
  "${SOURCE_ARGS[@]}"
  "${PGV_ARGS[@]}"
  --main-model "$BOARD_ROOT/rect_models/yolo11n_rk3566_int8_split_512x288.rknn"
  --specialist-model "$BOARD_ROOT/tl_check/tl_specialist_512x192_hybrid2.rknn"
  --detector-backend rknn \
  --display-backend gtk \
  --display-size 1024x576 \
  "${MODE_ARGS[@]}"
  "${FPGA_ARGS[@]}"
  "${HAZARD_ARGS[@]}"
  --output "$OUTPUT_DIR"
  --window-title "ADAS | LubanCat Competition Console"
)

if [[ "${FPGA_HUD:-0}" == "1" ]]; then
  exec sudo -n env \
    DISPLAY="$DISPLAY" \
    XAUTHORITY="$XAUTHORITY" \
    PYTHONPATH=/home/cat/.local/lib/python3.8/site-packages \
    "$PYTHON_BIN" "${RUNNER_ARGS[@]}"
fi

exec "$PYTHON_BIN" "${RUNNER_ARGS[@]}"
