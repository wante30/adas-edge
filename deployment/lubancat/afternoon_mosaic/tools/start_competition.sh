#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# Competition bench: real ARM -> FPGA commands, physical forward input Q4.
exec python3 "$ROOT/tools/test_session.py" start --build build-stage4 --label competition --ai-hz 0 --forward-quadrant 3 --control-mode HARDWARE --action-port 38262 --external-qml --authorize-control "$@"
