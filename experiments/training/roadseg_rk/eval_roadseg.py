"""Unified evaluation for Candidate-plus (4ch) and RoadSeg-RK (5ch) checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.training.fpga_road_seg.models import build_model as build_fpga
from .models_roadseg import build_roadseg_model, count_macs, parameter_count
from .roadseg_data import CLASSES5, RoadSegDataset
from .train_roadseg import CLASSES_A, Metrics, channel_layout, evaluate, merged_prob, supervised


def load_model(checkpoint, device):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    name = ck["config"]["model"]
    if name in ("candidate-plus", "candidate"):
        model = build_fpga(name)
        layout = "a"
        size = (ck["config"].get("width", 320), ck["config"].get("height", 192))
        names = CLASSES_A
    elif name == "candidate-plus-a":
        model = build_roadseg_model(name, classes=4, pretrained=False)
        layout = "a"
        size = (ck["config"]["width"], ck["config"]["height"])
        names = CLASSES_A
    else:
        layout = "five"
        model = build_roadseg_model(name, classes=5, pretrained=False, arch=ck["config"].get("arch", "small"),
                                    width=ck["config"].get("width_mult", 1.0))
        size = (ck["config"]["width"], ck["config"]["height"])
        names = CLASSES5
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device).eval()
    return model, ck, layout, size, names


@torch.no_grad()
def evaluate_detailed(model, dataset, device, layout, names, threshold=0.5, batch=16, workers=4):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    metrics = Metrics(len(names))
    merged = Metrics(1)
    rows = []
    total_loss = 0.0
    for x, y, v in loader:
        x, y, v = x.to(device), y.to(device), v.to(device)
        logits = model(x)
        sl, sy, sv = supervised(logits, y, v, layout)
        prob = sl.sigmoid()
        pred = (prob >= threshold)
        truth = sy > 0.5
        valid = sv > 0.5
        metrics.update_binary(pred.cpu().numpy(), truth.cpu().numpy(), valid.cpu().numpy())
        if layout == "five":
            mp = merged_prob(logits, layout)
            mt = ((y[:, 0] > 0.5) | (y[:, 1] > 0.5)).float()
            mv = ((v[:, 0] > 0.5) | (v[:, 1] > 0.5)).float()
        else:
            mp = logits[:, 0].sigmoid()
            mt = torch.maximum(y[:, 0], y[:, 1])
            mv = torch.maximum(v[:, 0], v[:, 1])
        merged.update_binary((mp >= threshold).cpu().numpy()[:, None], mt.cpu().numpy()[:, None], mv.cpu().numpy()[:, None])
        p_np, t_np, v_np = pred.cpu().numpy(), truth.cpu().numpy(), valid.cpu().numpy()
        for i in range(len(x)):
            item = {"id": dataset.rows[len(rows)]["id"], "source": dataset.rows[len(rows)]["source"]}
            per = {}
            for c, cls in enumerate(names):
                tp = int(np.sum(p_np[i, c] & t_np[i, c] & v_np[i, c]))
                fp = int(np.sum(p_np[i, c] & ~t_np[i, c] & v_np[i, c]))
                fn = int(np.sum(~p_np[i, c] & t_np[i, c] & v_np[i, c]))
                per[cls] = dict(iou=float(tp / (tp + fp + fn)) if tp + fp + fn else None,
                                f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
                                tp=tp, fp=fp, fn=fn)
            item["classes"] = per
            pm = (mp >= threshold)[i].cpu().numpy().astype(bool)
            tm = mt[i].cpu().numpy() > 0.5
            vm = mv[i].cpu().numpy() > 0.5
            inter = int(np.sum(pm & tm & vm))
            union = int(np.sum((pm | tm) & vm))
            item["drivable_merged"] = dict(iou=float(inter / union) if union else None,
                                           positives=int(tm.sum()), valid=int(vm.sum()))
            rows.append(item)
    report = dict(samples=len(dataset), threshold=threshold, names=names,
                  classes=metrics.report(names), merged=merged.report(["drivable_merged"])["drivable_merged"])
    return report, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-per-domain", type=int, default=0)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    args = p.parse_args()
    model, ck, layout, size, names = load_model(args.checkpoint, args.device)
    if args.width and args.height:
        size = (args.width, args.height)
    index = [json.loads(line) for line in Path(args.index).read_text(encoding="utf-8").splitlines() if line.strip()]
    domains = {}
    all_rows = {}
    for tag, source, split in (("bdd_internal_val", "bdd100k", "val"), ("bdd_official_val", "bdd100k", "test"),
                               ("rlmd_val", "rlmd", "val")):
        rows = [r for r in index if r["source"] == source and r["split"] == split]
        if not rows:
            print(f"skip {tag}: no rows", flush=True)
            continue
        if args.max_per_domain:
            rows = sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:args.max_per_domain]
        ds = RoadSegDataset(args.index, split, augment=False, size=size, seed=15)
        by_id = {r["id"]: r for r in ds.rows}
        ds.rows = [by_id[r["id"]] for r in rows]
        print(f"evaluate {tag}: {len(ds)}", flush=True)
        report, per_image = evaluate_detailed(model, ds, args.device, layout, names, args.threshold, args.batch, args.workers)
        domains[tag] = report
        all_rows[tag] = per_image
        print(json.dumps(report["classes"], indent=2, ensure_ascii=False), flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint=str(Path(args.checkpoint).resolve()), sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                    model=ck["config"]["model"], epoch=ck.get("epoch"), layout=layout, size=list(size),
                    params=parameter_count(model), macs=count_macs(model, (size[1], size[0])))
    (out / "metrics.json").write_text(json.dumps(dict(identity=identity, domains=domains), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for tag, rows in all_rows.items():
        (out / f"per_image_{tag}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(identity, indent=2), flush=True)


if __name__ == "__main__":
    main()
