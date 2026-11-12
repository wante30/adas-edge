#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
PYTHON=${PYTHON:-python3}
INPUT=${INPUT:?set INPUT to the real competition video}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/outputs/multi_signal_rk3566"}
MAIN_MODEL=${MAIN_MODEL:?set MAIN_MODEL to the INT8 B05N P2 RKNN file}
SPECIALIST_MODEL=${SPECIALIST_MODEL:-}

set -- "$PYTHON" "$ROOT/tools/rk3566_multi_signal_replay.py" \
  --input "$INPUT" --output-dir "$OUTPUT_DIR" \
  --main-model "$MAIN_MODEL" --main-hz "${MAIN_HZ:-10}" \
  --specialist-hz "${SPECIALIST_HZ:-2}" --conf-threshold "${CONF_THRESHOLD:-0.25}" \
  --specialist-conf-threshold "${SPECIALIST_CONF_THRESHOLD:-0.25}" \
  --nms-threshold "${NMS_THRESHOLD:-0.45}" \
  --specialist-nms-threshold "${SPECIALIST_NMS_THRESHOLD:-0.70}"
if [ -n "$SPECIALIST_MODEL" ]; then
  set -- "$@" --specialist-model "$SPECIALIST_MODEL"
fi
exec "$@"
