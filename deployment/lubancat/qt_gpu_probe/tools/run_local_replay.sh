#!/bin/sh
# Publish the official local quad replay as PGVF for the Qt source switch.
# Usage: tools/run_local_replay.sh [raw.rgb] [output.pgv]
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RAW=${1:-/home/cat/qt_gpu_probe_ai_20261019/local_quad.rgb}
OUT=${2:-/dev/shm/pgv_local/latest.pgv}
mkdir -p "$(dirname "$OUT")"
exec python3 "$ROOT/tools/pgv_local_publisher.py" \
  --raw "$RAW" --output "$OUT" --fps 30 --source-fps 30
