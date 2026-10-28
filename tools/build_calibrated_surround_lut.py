"""Generate the calibrated Open-3D-Surround-View LUT package on a PC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "adas_hmi_sim"))
from tools.adas_hmi_sim.backend.calibrated_surround import build_luts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-size", type=str, required=True, help="camera source WxH, e.g. 1920x1536")
    parser.add_argument("--output-size", type=str, default="320x320")
    parser.add_argument("--projection", choices=("bev", "bowl"), default="bowl")
    args = parser.parse_args()
    def pair(value: str) -> tuple[int, int]:
        width, height = value.lower().replace("×", "x").split("x", 1)
        return int(width), int(height)
    calibration = yaml.safe_load(args.calibration_yaml.read_text(encoding="utf-8"))
    manifest = build_luts(calibration, args.output, pair(args.source_size), pair(args.output_size), args.projection)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
