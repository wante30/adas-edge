#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT/fast_mosaic/capi"
make libp2capi_generic.so
mkdir -p "$ROOT/build-stage4"
cd "$ROOT/build-stage4"
qmake ../hmi/qt_gpu_probe.pro
make -j2
