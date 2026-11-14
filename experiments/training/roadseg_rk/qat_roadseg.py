"""Bounded QAT for RoadSeg-RK 5-channel models, exportable as QDQ ONNX."""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver, QConfig, QConfigMapping, disable_fake_quant,
    disable_observer, enable_fake_quant)
from torch.ao.quantization.quantize_fx import convert_to_reference_fx, prepare_qat_fx
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .eval_roadseg import load_model
from .roadseg_data import RoadSegDataset, _worker_init
from .train_roadseg import Metrics, evaluate, masked_loss, selection_score, supervised


def make_qconfig():
    return QConfig(
        activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
            dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False),
        weight=FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-128, quant_max=127,
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0, reduce_range=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--train-samples", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    random.seed(151); np.random.seed(151); torch.manual_seed(151)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model, ck, layout, size, names = load_model(args.checkpoint, "cpu")
    cfg = ck["config"]
    teacher = copy.deepcopy(model).eval().to(args.device)
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    qat = prepare_qat_fx(copy.deepcopy(model).train(), QConfigMapping().set_global(make_qconfig()),
                         (torch.zeros(1, 3, size[1], size[0]),)).to(args.device)
    qat.eval(); qat.apply(disable_fake_quant)
    calib = RoadSegDataset(args.index, "train", augment=False, size=size, seed=15)
    idx = sorted(range(len(calib)), key=lambda i: calib.rows[i]["id"])[:192]
    with torch.no_grad():
        for i in idx:
            x, _, _ = calib[i]
            qat(x[None].to(args.device))
    qat.apply(enable_fake_quant); qat.apply(disable_observer)

    train = RoadSegDataset(args.index, "train", augment=True, size=size, seed=15)
    counts = {}
    for r in train.rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    probs = cfg.get("source_probabilities", {"bdd100k": 0.8, "rlmd": 0.2})
    weights = [probs[r["source"]] / counts[r["source"]] for r in train.rows]
    sampler = WeightedRandomSampler(weights, min(args.train_samples, len(train)), replacement=True)
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=_worker_init,
                        prefetch_factor=3 if args.workers > 0 else None)
    val_bdd = RoadSegDataset(args.index, "val", size=size, seed=15)
    val_bdd.rows = [r for r in val_bdd.rows if r["source"] == "bdd100k"][:1500]
    val_rlmd = RoadSegDataset(args.index, "val", size=size, seed=15)
    val_rlmd.rows = [r for r in val_rlmd.rows if r["source"] == "rlmd"]
    val_loaders = {
        "bdd100k": DataLoader(val_bdd, batch_size=16, shuffle=False, num_workers=2, pin_memory=True),
        "rlmd": DataLoader(val_rlmd, batch_size=16, shuffle=False, num_workers=2, pin_memory=True),
    }
    optimizer = torch.optim.AdamW(qat.parameters(), lr=1e-5, weight_decay=1e-4)
    best_score, history = -1.0, []
    for epoch in range(args.epochs):
        qat.train(); qat.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        total = n = 0; t0 = time.perf_counter()
        for x, y, v in loader:
            x, y, v = x.to(args.device), y.to(args.device), v.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                ref = teacher(x)
            logits = qat(x)
            sl, sy, sv = supervised(logits, y, v, layout)
            loss = masked_loss(sl, sy, sv, cfg["pos_weight"], cfg.get("dice_weight", 1.0),
                               cfg.get("boundary_weight", 0.0), lane_start=1 if layout == "a" else 2)
            if layout == "five":
                reg = ((logits - ref).square() * v).sum() / v.sum().clamp_min(1)
                loss = loss + 0.2 * reg
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(x); n += len(x)
        domains = {k: evaluate(qat, dl, args.device, layout, names, cfg["threshold"]) for k, dl in val_loaders.items()}
        score = selection_score(domains, layout)
        history.append(dict(epoch=epoch + 1, loss=total / max(1, n), score=score, domains=domains,
                            seconds=time.perf_counter() - t0))
        print(f"qat epoch={epoch+1} loss={total/max(1,n):.4f} score={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            torch.save(dict(state_dict=qat.state_dict(), config=cfg, epoch=epoch + 1, score=score,
                            history=history), out / "qat.best.pt")
        (out / "qat_report.json").write_text(json.dumps(dict(best_score=best_score, history=history), indent=2) + "\n", encoding="utf-8")
    # QDQ export; reuse the audited reference-conv representation.
    from experiments.training.fpga_road_seg.qat_stage2 import exportable_reference
    best = torch.load(out / "qat.best.pt", map_location="cpu", weights_only=False)
    qat.load_state_dict(best["state_dict"]); qat.cpu().eval()
    reference = exportable_reference(qat)
    onnx_path = out / "qat.int8.qdq.onnx"
    torch.onnx.export(reference, torch.zeros(1, 3, size[1], size[0]), str(onnx_path),
                      input_names=["rgb"], output_names=["logits"], opset_version=17,
                      do_constant_folding=False, dynamo=False)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
