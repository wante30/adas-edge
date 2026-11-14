"""Save NHWC uint8 evaluation inputs for ONNX/RKNN parity checks."""
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
    p.add_argument("--bdd", type=int, default=16)
    p.add_argument("--rlmd", type=int, default=16)
    args = p.parse_args()
    rows = load_cache_index(args.index)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    bdd = sorted([r for r in rows if r["source"] == "bdd100k" and r["split"] == "val"],
                 key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:args.bdd]
    rlmd = [r for r in rows if r["source"] == "rlmd" and r["split"] == "val"]
    pos = [r for r in rlmd if r["positive"][4] > 0][:args.rlmd // 2]
    neg = [r for r in rlmd if r["positive"][4] == 0][:args.rlmd - len(pos)]
    selected = bdd + pos + neg
    manifest = []
    for i, row in enumerate(selected):
        bgr = cv2.imread(row["jpg"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        path = out / f"{i:03d}.npy"
        np.save(path, rgb[None].astype(np.uint8))
        manifest.append(dict(id=row["id"], source=row["source"], crosswalk_positive=row["positive"][4] > 0,
                             input=str(path.resolve()), shape=list(rgb.shape)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(samples=len(selected), bdd=len(bdd), rlmd=len(pos) + len(neg)), indent=2))


if __name__ == "__main__":
    main()
