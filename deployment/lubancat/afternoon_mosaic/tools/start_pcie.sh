#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec python3 "$ROOT/tools/test_session.py" start --build build-stage4 --label pcie --ai-hz 0 --action-port 38262 --external-qml
