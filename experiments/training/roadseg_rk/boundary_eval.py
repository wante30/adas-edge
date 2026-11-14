"""Boundary completeness and lane fragmentation metrics (2-pixel tolerance)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .eval_roadseg import load_model
from .roadseg_data import RoadSegDataset
from .train_roadseg import merged_prob, supervised


def boundary(mask, k=3):
    m = mask.float()
    dil = F.max_pool2d(m, k, 1, k // 2)
    ero = -F.max_pool2d(-m, k, 1, k // 2)
    return (dil - ero) > 0.5


def dilate(mask, k=5):
    return F.max_pool2d(mask.float(), k, 1, k // 2) > 0.5


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-images", type=int, default=500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    model, ck, layout, size, names = load_model(args.checkpoint, args.device)
    ds = RoadSegDataset(args.index, "test", size=size, seed=15)
    ds.rows = [r for r in ds.rows if r["source"] == "bdd100k"][:args.max_images]
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    stats = {k: np.zeros(4, np.int64) for k in ("drivable", "solid", "dashed")}
    components = {k: [] for k in ("solid", "dashed")}
    for x, y, v in loader:
        x, y, v = x.to(args.device), y.to(args.device), v.to(args.device)
        logits = model(x)
        sl, sy, sv = supervised(logits, y, v, layout)
        if layout == "five":
            masks = {"drivable": merged_prob(logits, layout) >= 0.5,
                     "solid": sl[:, 2].sigmoid() >= 0.5,
                     "dashed": sl[:, 3].sigmoid() >= 0.5}
            gts = {"drivable": ((y[:, 0] > 0.5) | (y[:, 1] > 0.5)),
                   "solid": y[:, 2] > 0.5, "dashed": y[:, 3] > 0.5}
            vals = {"drivable": ((v[:, 0] > 0.5) | (v[:, 1] > 0.5)),
                    "solid": v[:, 2] > 0.5, "dashed": v[:, 3] > 0.5}
        else:
            masks = {"drivable": sl[:, 0].sigmoid() >= 0.5,
                     "solid": sl[:, 1].sigmoid() >= 0.5,
                     "dashed": sl[:, 2].sigmoid() >= 0.5}
            gts = {"drivable": sy[:, 0] > 0.5, "solid": sy[:, 1] > 0.5, "dashed": sy[:, 2] > 0.5}
            vals = {"drivable": sv[:, 0] > 0.5, "solid": sv[:, 1] > 0.5, "dashed": sv[:, 2] > 0.5}
        for key in ("drivable", "solid", "dashed"):
            pred = masks[key]
            gt = gts[key]
            valid = vals[key]
            pb = boundary(pred) & valid
            gb = boundary(gt) & valid
            tp = int((pb & dilate(gb)).sum())
            fp = int((pb & ~dilate(gb)).sum())
            fn = int((gb & ~dilate(pb)).sum())
            stats[key] += np.array([tp, fp, fn, int(valid.sum())], np.int64)
            if key in components:
                for i in range(pred.shape[0]):
                    m = (pred[i].cpu().numpy().astype(np.uint8))
                    if valid[i].sum() > 0:
                        n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
                        area = sum(st[j, cv2.CC_STAT_AREA] for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= 5)
                        count = sum(1 for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= 5)
                        if area > 0:
                            components[key].append(count)
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "model": ck["config"]["model"],
              "images": len(ds), "boundary_tolerance_px": 2, "classes": {}}
    for key, (tp, fp, fn, valid) in stats.items():
        tp, fp, fn, valid = int(tp), int(fp), int(fn), int(valid)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        report["classes"][key] = dict(boundary_tp=tp, boundary_fp=fp, boundary_fn=fn,
                                      boundary_precision=precision, boundary_recall=recall,
                                      boundary_f1=2 * precision * recall / max(1e-9, precision + recall))
    for key, values in components.items():
        report["classes"][key]["mean_components_per_image"] = float(np.mean(values)) if values else None
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
