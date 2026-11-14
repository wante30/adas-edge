"""Per-class threshold sweep on internal validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .eval_roadseg import load_model
from .roadseg_data import RoadSegDataset
from .train_roadseg import Metrics, merged_prob, supervised


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-bdd", type=int, default=2000)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    model, ck, layout, size, names = load_model(args.checkpoint, args.device)
    ds = RoadSegDataset(args.index, "val", size=size, seed=15)
    ds.rows = [r for r in ds.rows if r["source"] == "bdd100k"][:args.max_bdd]
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    metrics = {t: Metrics(len(names)) for t in thresholds}
    merged = {t: Metrics(1) for t in thresholds}
    for x, y, v in loader:
        x, y, v = x.to(args.device), y.to(args.device), v.to(args.device)
        logits = model(x)
        sl, sy, sv = supervised(logits, y, v, layout)
        prob = sl.sigmoid()
        if layout == "five":
            mp = merged_prob(logits, layout)
            mt = ((y[:, 0] > 0.5) | (y[:, 1] > 0.5)).float()
            mv = ((v[:, 0] > 0.5) | (v[:, 1] > 0.5)).float()
        else:
            mp = logits[:, 0].sigmoid()
            mt = torch.maximum(y[:, 0], y[:, 1])
            mv = torch.maximum(v[:, 0], v[:, 1])
        for t in thresholds:
            metrics[t].update_binary((prob >= t).cpu().numpy(), (sy > 0.5).cpu().numpy(), (sv > 0.5).cpu().numpy())
            merged[t].update_binary((mp >= t).cpu().numpy()[:, None], mt.cpu().numpy()[:, None], mv.cpu().numpy()[:, None])
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "model": ck["config"]["model"],
              "samples": len(ds), "thresholds": {}}
    for t in thresholds:
        report["thresholds"][str(t)] = dict(classes=metrics[t].report(names),
                                            merged=merged[t].report(["drivable_merged"])["drivable_merged"])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for t in thresholds:
        c = report["thresholds"][str(t)]["classes"]
        print(t, "merged", report["thresholds"][str(t)]["merged"]["iou"],
              "solid_f1", c["solid_lane"]["f1"], "dashed_f1", c["dashed_lane"]["f1"])


if __name__ == "__main__":
    main()
