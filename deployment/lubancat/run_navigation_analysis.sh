#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PGV_FILE="${PGV_FILE:-/dev/shm/pgv_live_codex_20261018_postreboot/latest.pgv}"

if [[ ! -r "$PGV_FILE" ]]; then
  echo "PCIe video stream is unavailable: $PGV_FILE" >&2
  exit 2
fi

export PGV_FILE
export FPGA_HUD="${FPGA_HUD:-1}"
export HAZARDS="${HAZARDS:-1}"
export PGV_QUADRANT="${PGV_QUADRANT:-3}"
export PGV_NAVIGATION_QUADRANT="${PGV_NAVIGATION_QUADRANT:-1}"

exec "$SCRIPT_DIR/run_competition_console.sh"
