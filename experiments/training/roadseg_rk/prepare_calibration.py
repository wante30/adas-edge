"""Build RKNN INT8 calibration tensors from real TRAIN cache samples only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .roadseg_data import load_cache_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bdd", type=int, default=96)
    p.add_argument("--rlmd-positive", type=int, default=48)
    p.add_argument("--rlmd-negative", type=int, default=48)
    p.add_argument("--width", type=int, default=416)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--path-style", choices=("windows", "wsl"), default="windows")
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_cache_index(args.index) if r["split"] == "train"]
    bdd = sorted([r for r in rows if r["source"] == "bdd100k"],
                 key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:args.bdd]
    rlmd = [r for r in rows if r["source"] == "rlmd"]
    pos = sorted([r for r in rlmd if r["positive"][4] > 0], key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:args.rlmd_positive]
    neg = sorted([r for r in rlmd if r["positive"][4] == 0], key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:args.rlmd_negative]
    selected = bdd + pos + neg
    if len(selected) < 1:
        raise SystemExit("no calibration rows")
    lines = []
    manifest = []
    w, h = args.width, args.height
    def display_path(path):
        resolved = path.resolve()
        if args.path_style == "wsl":
            text = str(resolved).replace("\\", "/")
            if len(text) > 1 and text[1] == ":":
                return "/mnt/" + text[0].lower() + text[2:]
            return text
        return str(resolved)
    for i, row in enumerate(selected):
        bgr = cv2.imread(row["jpg"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["jpg"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (w, h):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        tensor = np.transpose(rgb, (2, 0, 1))[None].astype(np.uint8)  # NCHW, required by RKNN .npy loader
        path = out / f"{i:04d}.npy"
        np.save(path, tensor)
        lines.append(display_path(path))
        manifest.append(dict(id=row["id"], source=row["source"], split=row["split"], crosswalk_positive=row["positive"][4] > 0,
                             path=display_path(path)))
    (out / "dataset.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "calibration_manifest.json").write_text(json.dumps(dict(
        samples=len(selected), bdd=len(bdd), rlmd_positive=len(pos), rlmd_negative=len(neg),
        shape=[1, 3, h, w], layout="NCHW uint8 RGB, already letterboxed",
        source_index=str(Path(args.index).resolve()), rows=manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(samples=len(selected), shape=[1, 3, h, w]), indent=2))


if __name__ == "__main__":
    main()
