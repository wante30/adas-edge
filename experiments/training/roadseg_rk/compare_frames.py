"""Same-frame GT / Candidate-plus / RoadSeg-RK comparison sheets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .eval_roadseg import load_model
from .postprocess_np import compose, preprocess
from .visualize_roadseg import infer

GT_COLORS = [(0, 200, 0), (0, 120, 120), (0, 0, 235), (235, 220, 80), (210, 90, 210)]


def gt_overlay(image_bgr, y, v, alpha=0.45):
    h, w = image_bgr.shape[:2]
    small = cv2.resize(image_bgr, (416, 256), interpolation=cv2.INTER_LINEAR)
    ov = small.copy()
    for c in range(5):
        m = (y[c] > 0.5) & (v[c] > 0.5)
        ov[m] = GT_COLORS[c]
    vis = cv2.addWeighted(small, 1 - alpha, ov, alpha, 0)
    return cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (max(120, 14 * len(text)), 30), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--checkpoint", action="append", required=True)
    p.add_argument("--label", action="append", default=[])
    p.add_argument("--ids", nargs="*", default=[])
    p.add_argument("--id-file")
    p.add_argument("--source")
    p.add_argument("--split")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    rows = [json.loads(line) for line in Path(args.index).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {r["id"]: r for r in rows}
    if args.id_file:
        selected = [by_id[i] for i in Path(args.id_file).read_text(encoding="utf-8").splitlines() if i.strip()]
    elif args.ids:
        selected = [by_id[i] for i in args.ids]
    else:
        pool = [r for r in rows if (not args.source or r["source"] == args.source) and (not args.split or r["split"] == args.split)]
        selected = pool[:args.n]
    models = []
    for checkpoint in args.checkpoint:
        model, ck, layout, size, names = load_model(checkpoint, args.device)
        models.append((model, layout, size, ck["config"]["model"]))
    labels = args.label or [m[3] for m in models]
    panels = []
    for r in selected:
        bgr = cv2.imread(r["jpg"])
        with np.load(r["masks"], allow_pickle=False) as m:
            bits = m["bits"]
        a = np.unpackbits(bits, axis=-1, count=r["shape"][1], bitorder="little")
        y, v = a[:5].astype(np.float32), a[5:].astype(np.float32)
        row = [label(bgr, r["id"])]
        row.append(label(gt_overlay(bgr, y, v), "GT"))
        for (model, layout, size, _), text in zip(models, labels):
            rgb, _ = preprocess(bgr, size)
            probs = infer(model, rgb, args.device, layout)
            vis, _ = compose(bgr, probs, 0.5, False, False)
            row.append(label(vis, text))
        panels.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(panels, axis=0)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out} shape={sheet.shape}")


if __name__ == "__main__":
    main()
