#!/bin/sh
set -eu

ROOT="${ADAS_EDGE_ROOT:-/home/cat/competition_cockpit_port_20261018}"
SOURCE="$ROOT/competition_ai/rga_preprocess.cpp"
OUTPUT="$ROOT/competition_ai/librga_preprocess.so"

test -c /dev/rga
pkg-config --exists librga
g++ -O3 -std=gnu++14 -fPIC -shared \
    $(pkg-config --cflags librga) "$SOURCE" \
    -o "$OUTPUT" $(pkg-config --libs librga)
printf 'built %s\n' "$OUTPUT"
