#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RAW=${1:-/home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb}
exec python3 "$ROOT/tools/test_session.py" start --build build-stage4 --label local_replay --ai-hz 0 --control-mode DRY_RUN --external-qml --raw-replay "$RAW"
