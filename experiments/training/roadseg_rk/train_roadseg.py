"""Train RoadSeg-RK candidates on the cached 5-channel BDD+RLMD data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .models_roadseg import build_roadseg_model, count_macs, parameter_count
from .roadseg_data import CLASSES5, RoadSegDataset, _worker_init

CLASSES_A = ("drivable_area", "solid_lane", "dashed_lane", "crosswalk")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def channel_layout(model_name):
    return "a" if model_name == "candidate-plus-a" else "five"


def supervised(logits, y, v, layout):
    if layout == "a":
        merged = torch.maximum(y[:, 0], y[:, 1])
        mvalid = torch.maximum(v[:, 0], v[:, 1])
        return logits, torch.stack([merged, y[:, 2], y[:, 3], y[:, 4]], 1), \
            torch.stack([mvalid, v[:, 2], v[:, 3], v[:, 4]], 1)
    return logits, y, v


def merged_prob(logits, layout):
    if layout == "a":
        return logits[:, 0].sigmoid()
    return 1 - (1 - logits[:, 0].sigmoid()) * (1 - logits[:, 1].sigmoid())


def masked_loss(logits, target, valid, pos_weight, dice_weight=1.0, boundary_weight=0.0, lane_start=2):
    counts = valid.sum((0, 2, 3))
    active = counts > 0
    if not active.any():
        return logits.sum() * 0
    weights = logits.new_tensor(pos_weight)[:, None, None]
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weights, reduction="none")
    if boundary_weight > 0:
        # Soft boundary band from the target, without changing label semantics.
        dilate = F.max_pool2d(target, 3, 1, 1)
        erode = -F.max_pool2d(-target, 3, 1, 1)
        band = (dilate - erode).clamp(0, 1)
        weight = torch.ones_like(target)
        weight[:, lane_start:] = 1 + boundary_weight * band[:, lane_start:]
        bce = bce * weight
    bce = (bce * valid).sum((0, 2, 3)) / counts.clamp_min(1)
    prob = logits.sigmoid() * valid
    truth = target * valid
    dice = 1 - (2 * (prob * truth).sum((0, 2, 3)) + 1) / (prob.sum((0, 2, 3)) + truth.sum((0, 2, 3)) + 1)
    return (bce + dice_weight * dice)[active].mean()


class Metrics:
    def __init__(self, channels):
        self.tp = np.zeros(channels, np.int64)
        self.fp = self.tp.copy(); self.fn = self.tp.copy(); self.tn = self.tp.copy()
        self.valid = self.tp.copy()

    def update_binary(self, pred, truth, valid):
        p = np.asarray(pred).astype(bool)
        t = np.asarray(truth) > 0.5
        v = np.asarray(valid) > 0.5
        self.tp += np.sum(p & t & v, (0, 2, 3)).astype(np.int64)
        self.fp += np.sum(p & ~t & v, (0, 2, 3)).astype(np.int64)
        self.fn += np.sum(~p & t & v, (0, 2, 3)).astype(np.int64)
        self.tn += np.sum(~p & ~t & v, (0, 2, 3)).astype(np.int64)
        self.valid += np.sum(v, (0, 2, 3)).astype(np.int64)

    def report(self, names):
        def ratio(a, b):
            return float(a / b) if b else None
        return {n: dict(iou=ratio(a, a + b + c), f1=ratio(2 * a, 2 * a + b + c),
                        recall=ratio(a, a + c), precision=ratio(a, a + b),
                        tp=int(a), fp=int(b), fn=int(c), tn=int(d), valid_pixels=int(v))
                for n, a, b, c, d, v in zip(names, self.tp, self.fp, self.fn, self.tn, self.valid)}


@torch.no_grad()
def evaluate(model, loader, device, layout, names, threshold=0.5):
    model.eval()
    n = len(names)
    metrics = Metrics(n)
    merged = Metrics(1)
    loss_sum = total = 0
    for x, y, v in loader:
        x, y, v = x.to(device), y.to(device), v.to(device)
        logits = model(x)
        sl, sy, sv = supervised(logits, y, v, layout)
        loss_sum += float(masked_loss(sl, sy, sv, [1.0] * sl.shape[1])) * len(x)
        total += len(x)
        prob = sl.sigmoid()
        metrics.update_binary((prob >= threshold).cpu().numpy(), (sy > 0.5).cpu().numpy(), (sv > 0.5).cpu().numpy())
        if layout == "five":
            mp = merged_prob(logits, layout)
            mt = ((y[:, 0] > 0.5) | (y[:, 1] > 0.5)).float()
            mv = ((v[:, 0] > 0.5) | (v[:, 1] > 0.5)).float()
        else:
            mp = logits[:, 0].sigmoid()
            mt = torch.maximum(y[:, 0], y[:, 1])
            mv = torch.maximum(v[:, 0], v[:, 1])
        merged.update_binary((mp >= threshold).cpu().numpy()[:, None], mt.cpu().numpy()[:, None], mv.cpu().numpy()[:, None])
    return dict(loss=loss_sum / max(total, 1), samples=total,
                classes=metrics.report(names), merged=merged.report(["drivable_merged"])["drivable_merged"],
                threshold=threshold)


def merged_union_loss(logits, y, v):
    merged_logit = torch.logsumexp(torch.stack([logits[:, 0], logits[:, 1]], 1), 1)
    p = merged_logit.sigmoid()
    target = torch.maximum(y[:, 0], y[:, 1])
    valid = torch.maximum(v[:, 0], v[:, 1])
    bce = (F.binary_cross_entropy_with_logits(merged_logit, target, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    dice = 1 - (2 * (p * target * valid).sum() + 1) / ((p * valid).sum() + (target * valid).sum() + 1)
    return bce + dice


def selection_score(domains, layout):
    b = domains["bdd100k"]["classes"]
    merged = domains["bdd100k"]["merged"]["iou"] or 0
    solid = b["solid_lane"]["f1"] or 0
    dashed = b["dashed_lane"]["f1"] or 0
    cross = domains["rlmd"]["classes"]["crosswalk"]["iou"] or 0
    if layout == "a":
        return 0.40 * merged + 0.25 * solid + 0.15 * dashed + 0.20 * cross
    return 0.35 * merged + 0.22 * solid + 0.18 * dashed + 0.25 * cross


def stratified_subset(rows, limit, seed):
    if not limit or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    for values in by_source.values():
        rng.shuffle(values)
    total = len(rows)
    out = []
    for source, values in sorted(by_source.items()):
        want = int(round(limit * len(values) / total))
        out.extend(values[:want])
    # Fill any shortfall deterministically without breaking source balance too far.
    if len(out) < limit:
        chosen = {r["id"] for r in out}
        pool = [r for r in rows if r["id"] not in chosen]
        rng.shuffle(pool)
        out.extend(pool[:limit - len(out)])
    return out[:limit]


def save_curve(path, history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([e["epoch"] for e in history], [e["train_loss"] for e in history], label="train")
    axes[0].plot([e["epoch"] for e in history], [e["val_loss"] for e in history], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=.3)
    for key, label in (("merged_iou", "drivable IoU"), ("solid_f1", "solid F1"), ("dashed_f1", "dashed F1"), ("crosswalk_iou", "crosswalk IoU")):
        axes[1].plot([e["epoch"] for e in history], [e[key] for e in history], label=label)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("metric"); axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True, help="cache_index.jsonl")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--init")
    p.add_argument("--init-partial", action="store_true", help="load matching keys only (new decoder heads)")
    p.add_argument("--teacher")
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    layout = channel_layout(cfg["model"])
    size = (cfg["width"], cfg["height"])
    names = CLASSES5 if layout == "five" else CLASSES_A
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not args.resume:
        raise SystemExit("nonempty output requires --resume or a new directory")

    all_rows = [json.loads(line) for line in Path(args.index).read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_bdd = [r for r in all_rows if r["split"] == "val" and r["source"] == "bdd100k"]
    val_rlmd = [r for r in all_rows if r["split"] == "val" and r["source"] == "rlmd"]
    if args.train_limit:
        train_rows = stratified_subset(train_rows, args.train_limit, seed)
    val_bdd = sorted(val_bdd, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    if args.val_limit:
        val_bdd = val_bdd[:args.val_limit]

    def domain_loader(rows, split, augment=False, batch=16, workers=2, sampler=None):
        ds = RoadSegDataset(args.index, split, augment=augment, size=size, seed=seed)
        by_id = {r["id"]: r for r in ds.rows}
        ds.rows = [by_id[r["id"]] for r in rows]
        sampler_obj = None
        if sampler == "weighted":
            counts = Counter(r["source"] for r in rows)
            probs = cfg.get("source_probabilities", {"bdd100k": 0.8, "rlmd": 0.2})
            weights = [probs[r["source"]] / counts[r["source"]] for r in rows]
            samples = min(cfg.get("epoch_samples", len(rows)), len(rows))
            sampler_obj = WeightedRandomSampler(weights, samples, replacement=True)
        def worker_init(worker_id):
            info = torch.utils.data.get_worker_info()
            info.dataset.rng.seed(seed * 1000 + worker_id)
        return DataLoader(ds, batch_size=batch, shuffle=sampler_obj is None, sampler=sampler_obj,
                          num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                          prefetch_factor=3 if workers > 0 else None, worker_init_fn=_worker_init)

    train_loader = domain_loader(train_rows, "train", augment=True, batch=cfg["batch_size"], workers=args.workers, sampler="weighted")
    bdd_loader = domain_loader(val_bdd, "val", batch=16, workers=2)
    rlmd_loader = domain_loader(val_rlmd, "val", batch=16, workers=2)
    print(f"train={len(train_rows)} bdd_val={len(val_bdd)} rlmd_val={len(val_rlmd)}", flush=True)

    model = build_roadseg_model(cfg["model"], classes=5 if layout == "five" else 4,
                                pretrained=cfg.get("pretrained", False), arch=cfg.get("arch", "small"),
                                width=cfg.get("width_mult", 1.0)).to(args.device)
    if args.init:
        ck = torch.load(args.init, map_location=args.device, weights_only=False)
        state = ck["state_dict"]
        if cfg["model"] == "candidate-plus-a":
            state = {k.removeprefix("net."): v for k, v in state.items()}
            model.net.load_state_dict(state, strict=True)
        elif cfg["model"] == "candidate-plus-5":
            old = {k.removeprefix("net."): v for k, v in state.items()}
            model.net.load_state_dict({k: v for k, v in old.items() if not k.startswith("head.")}, strict=False)
            with torch.no_grad():
                for new_idx, old_idx in ((0, 0), (2, 1), (3, 2), (4, 3)):
                    model.net.head.weight[new_idx].copy_(old["head.weight"][old_idx])
                    model.net.head.bias[new_idx].copy_(old["head.bias"][old_idx])
                model.net.head.weight[1].copy_(old["head.weight"][0])
                model.net.head.bias[1].copy_(old["head.bias"][0])
            print("candidate-plus-5 head transferred: 4ch -> 5ch", flush=True)
        elif args.init_partial:
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"partial init missing={list(missing)[:12]} unexpected={list(unexpected)[:12]}", flush=True)
        else:
            model.load_state_dict(state, strict=True)
        print(f"initialized from {args.init} epoch={ck.get('epoch')}", flush=True)
    teacher = None
    if args.teacher:
        from experiments.training.fpga_road_seg.models import build_model as build_fpga
        tck = torch.load(args.teacher, map_location=args.device, weights_only=False)
        teacher = build_fpga(tck["config"]["model"]).to(args.device).eval()
        teacher.load_state_dict(tck["state_dict"], strict=True)
        teacher.requires_grad_(False)
        print(f"teacher loaded: {tck['config']['model']} epoch={tck.get('epoch')}", flush=True)

    params = parameter_count(model)
    macs = count_macs(model, size)
    print(f"params={params} macs={macs}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=cfg["lr"] * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and args.device.startswith("cuda")))
    history, best_score, stale, start_epoch = [], -1.0, 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        if ck.get("optimizer"): optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"): scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
        history = ck.get("history", [])
        best_score = ck.get("best_score", -1.0)
        stale = ck.get("stale", 0)
        start_epoch = ck.get("epoch", 0)
        print(f"resumed epoch={start_epoch} best={best_score}", flush=True)

    identity = dict(index_sha256=sha256(args.index), config=cfg, params=params, macs=macs,
                    model=cfg["model"], layout=layout, size=list(size))
    write_json(out / "model_identity.json", identity)
    csv_path = out / "metrics.csv"
    if not csv_path.exists():
        csv_path.write_text("epoch,train_loss,val_loss,merged_iou,merged_f1,solid_f1,dashed_f1,crosswalk_iou,crosswalk_f1,score,seconds\n", encoding="utf-8")

    for epoch in range(start_epoch, epochs):
        model.train()
        loss_sum = total = 0
        t0 = time.perf_counter()
        for x, y, v in train_loader:
            x, y, v = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True), v.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = model(x)
                sl, sy, sv = supervised(logits, y, v, layout)
                loss = masked_loss(sl, sy, sv, cfg["pos_weight"], cfg.get("dice_weight", 1.0),
                                   cfg.get("boundary_weight", 0.0), lane_start=1 if layout == "a" else 2)
                if layout == "five" and cfg.get("merged_weight", 0) > 0:
                    loss = loss + cfg["merged_weight"] * merged_union_loss(logits, y, v)
                if teacher is not None and layout == "five" and cfg.get("distill_weight", 0) > 0:
                    with torch.no_grad():
                        tin = F.interpolate(x, size=(192, 320), mode="bilinear", align_corners=False)
                        tlogit = teacher(tin)
                        tprob = F.interpolate(tlogit.sigmoid(), size=x.shape[-2:], mode="bilinear",
                                              align_corners=False)
                    if tlogit.shape[1] >= 4:
                        smerged = torch.logsumexp(torch.stack([logits[:, 0], logits[:, 1]], 1), 1)
                        student_teacher = torch.stack([smerged, logits[:, 2], logits[:, 3], logits[:, 4]], 1)
                        kd = F.binary_cross_entropy_with_logits(student_teacher, tprob, reduction="none")
                        known = torch.stack([torch.maximum(v[:, 0], v[:, 1]), v[:, 2], v[:, 3], v[:, 4]], 1)
                    else:
                        smerged = torch.logsumexp(torch.stack([logits[:, 0], logits[:, 1]], 1), 1)
                        kd = F.binary_cross_entropy_with_logits(smerged, tprob[:, 0], reduction="none")
                        known = torch.maximum(v[:, 0], v[:, 1])
                    loss = loss + cfg["distill_weight"] * (kd * known).sum() / known.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach()) * len(x); total += len(x)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        bdd = evaluate(model, bdd_loader, args.device, layout, names, cfg["threshold"])
        rlmd = evaluate(model, rlmd_loader, args.device, layout, names, cfg["threshold"])
        domains = {"bdd100k": bdd, "rlmd": rlmd}
        score = selection_score(domains, layout)
        improved = score > best_score + cfg.get("min_delta", 0.001)
        if improved:
            best_score = score; stale = 0
        else:
            stale += 1
        b = bdd["classes"]
        entry = dict(epoch=epoch + 1, train_loss=loss_sum / max(total, 1), val_loss=(bdd["loss"] + rlmd["loss"]) / 2,
                     train_seconds=seconds, train_img_s=total / seconds,
                     merged_iou=bdd["merged"]["iou"], merged_f1=bdd["merged"]["f1"],
                     solid_f1=b["solid_lane"]["f1"], solid_iou=b["solid_lane"]["iou"],
                     dashed_f1=b["dashed_lane"]["f1"], dashed_iou=b["dashed_lane"]["iou"],
                     crosswalk_iou=rlmd["classes"]["crosswalk"]["iou"], crosswalk_f1=rlmd["classes"]["crosswalk"]["f1"],
                     score=score, bdd=bdd, rlmd=rlmd)
        history.append(entry)
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([entry["epoch"], entry["train_loss"], entry["val_loss"], entry["merged_iou"], entry["merged_f1"],
                             entry["solid_f1"], entry["dashed_f1"], entry["crosswalk_iou"], entry["crosswalk_f1"],
                             entry["score"], entry["train_seconds"]])
        print(f"epoch={epoch+1} loss={entry['train_loss']:.4f} merged={entry['merged_iou']:.4f} "
              f"solid={entry['solid_f1']:.4f} dashed={entry['dashed_f1']:.4f} cross={entry['crosswalk_iou']:.4f} "
              f"score={score:.4f} time={seconds:.1f}s", flush=True)
        ck = dict(state_dict=model.state_dict(), config=cfg, epoch=epoch + 1, history=history,
                  optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                  best_score=best_score, stale=stale, identity=identity)
        torch.save(ck, out / "last.tmp.pt"); os.replace(out / "last.tmp.pt", out / "last.pt")
        if improved:
            torch.save(ck, out / "best.tmp.pt"); os.replace(out / "best.tmp.pt", out / "best.pt")
        write_json(out / "report.json", dict(identity=identity, epoch=epoch + 1, best_score=best_score,
                                             stale=stale, device=args.device,
                                             gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                                             amp=bool(scaler.is_enabled()), history=history))
        save_curve(out / "training_curves.png", history)
        scheduler.step()
        if epoch + 1 >= cfg.get("min_epochs", 8) and stale >= cfg.get("patience", 4):
            print(f"early_stop epoch={epoch+1} stale={stale}", flush=True)
            break
    write_json(out / "training_complete.json", dict(epochs=len(history), best_score=best_score,
                                                    finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


if __name__ == "__main__":
    main()
