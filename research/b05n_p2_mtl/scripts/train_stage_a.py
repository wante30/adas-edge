"""Stage A: freeze B05N P2 detector, train the shared segmentation decoder only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtl.data import BDDMTLDataset, collate  # noqa: E402
from mtl.losses import (  # noqa: E402
    batch_stats, drivable_loss, lane_category_loss, lane_style_loss,
)
from mtl.model import build_model  # noqa: E402


def seg_losses(seg, batch, lane_weights):
    h, w = batch["lane_cat"].shape[-2:]
    logits = {k: F.interpolate(v, size=(h, w), mode="bilinear", align_corners=False)
              for k, v in seg.items()}

    lc_ce, lc_dice, lc_fg, lc_tv = lane_category_loss(logits["lane_cat"], batch["lane_cat"], lane_weights)
    ls_ce, ls_dice = lane_style_loss(logits["lane_style"], batch["lane_style"])
    dr_ce, dr_dice = drivable_loss(logits["drivable"], batch["drivable"])

    lane_w = batch["has_lane"].float().mean().clamp(min=1e-6)
    drv_w = batch["has_drivable"].float().mean().clamp(min=1e-6)
    lane = (0.6 * lc_ce + 1.5 * lc_dice + 2.0 * lc_fg + 2.0 * lc_tv) * lane_w
    style = (ls_ce + ls_dice) * lane_w
    drv = (dr_ce + dr_dice) * drv_w

    total = lane + 0.5 * style + drv
    parts = {
        "loss": total.detach(),
        "lane_ce": lc_ce.detach(), "lane_dice": lc_dice.detach(), "lane_fg_dice": lc_fg.detach(), "lane_tversky": lc_tv.detach(),
        "style_ce": ls_ce.detach(), "style_dice": ls_dice.detach(),
        "drv_ce": dr_ce.detach(), "drv_dice": dr_dice.detach(),
    }
    return total, parts, logits


@torch.no_grad()
def evaluate(model, loader, lane_weights):
    model.eval()
    agg, n = {}, 0
    for batch in loader:
        batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=True):
            seg, _, _ = model(batch["image"], run_det=False)
            total, parts, _ = seg_losses(seg, batch, lane_weights)
            stats = batch_stats(seg, batch)
        bs = batch["image"].shape[0]
        n += bs
        for k, v in {**parts, **stats, "total": total}.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bs
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/bdd100k")
    ap.add_argument("--ckpt", default="weights/b05n_p2_multiclass.pt")
    ap.add_argument("--out", default="runs/stageA")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mid", type=int, default=64)
    ap.add_argument("--head-ch", type=int, default=64)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = BDDMTLDataset(args.root, "train")
    if args.limit_train:
        train_ds = Subset(train_ds, list(range(min(args.limit_train, len(train_ds)))))
    val_ds = BDDMTLDataset(args.root, "val")
    val_ds = Subset(val_ds, list(range(min(args.limit_val, len(val_ds)))))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, collate_fn=collate, pin_memory=True)

    model = build_model(args.ckpt, device="cuda", mid=args.mid, head_ch=args.head_ch)
    if args.resume:
        ck = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"resumed from {args.resume} (epoch {ck.get('epoch')})", flush=True)
    n_seg = sum(p.numel() for p in model.seg_parameters())
    print(f"det frozen | seg decoder params: {n_seg/1e6:.3f}M | det taps {model.tap_idx}", flush=True)

    lane_weights = torch.tensor([25, 25, 25, 25, 25, 25, 25, 25, 1.0], device="cuda")
    opt = torch.optim.AdamW(model.seg_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.epochs * steps_per_epoch, 1),
        pct_start=0.1, div_factor=10, final_div_factor=50)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    log_path = out / "train_log.csv"
    log_f = open(log_path, "a", newline="")
    writer = csv.writer(log_f)
    if log_path.stat().st_size == 0:
        writer.writerow(["epoch", "step", "lr", "loss", "lane_ce", "lane_dice", "lane_fg_dice",
                         "style_ce", "style_dice", "drv_ce", "drv_dice", "lane_fg_iou",
                         "crosswalk_iou", "style_acc", "drivable_miou"])

    best_score = -1.0
    best_loss = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, run_stats, n_seen = 0.0, {}, 0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=True):
                seg, _, _ = model(batch["image"], run_det=False)
                total, parts, logits = seg_losses(seg, batch, lane_weights)

            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.seg_parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            bs = batch["image"].shape[0]
            n_seen += bs
            run += float(total) * bs
            for k, v in parts.items():
                run_stats[k] = run_stats.get(k, 0.0) + float(v) * bs

            if step % 20 == 0:
                with torch.no_grad():
                    stats = batch_stats(seg, batch)
                stats_s = " ".join(f"{k}={v:.3f}" for k, v in stats.items())
                print(f"e{epoch} {step}/{steps_per_epoch} loss={run/n_seen:.4f} "
                      f"lane_ce={run_stats['lane_ce']/n_seen:.3f} fg={run_stats['lane_fg_dice']/n_seen:.3f} "
                      f"style_ce={run_stats['style_ce']/n_seen:.3f} "
                      f"drv_ce={run_stats['drv_ce']/n_seen:.3f} {stats_s} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
                writer.writerow([epoch, step, f"{sched.get_last_lr()[0]:.6g}",
                                 f"{run/n_seen:.5f}"] +
                                [f"{run_stats[k]/n_seen:.5f}" for k in
                                 ("lane_ce", "lane_dice", "lane_fg_dice", "style_ce", "style_dice",
                                  "drv_ce", "drv_dice")] +
                                [f"{stats['lane_fg_iou']:.5f}", f"{stats['crosswalk_iou']:.5f}",
                                 f"{stats['style_acc']:.5f}", f"{stats['drivable_miou']:.5f}"])
                log_f.flush()

        ckpt = {
            "model": model.state_dict(),
            "arch": {"mid": args.mid, "head_ch": args.head_ch, "tap_idx": model.tap_idx},
            "source": args.ckpt,
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(ckpt, out / "mtl_last.pt")
        print(f"[epoch {epoch}] saved {out/'mtl_last.pt'} elapsed={(time.time()-t0)/60:.1f}min", flush=True)

        if epoch % args.val_every == 0 and len(val_loader):
            m = evaluate(model, val_loader, lane_weights)
            score = (m["lane_fg_iou"] + m["crosswalk_iou"]
                     + 0.5 * m["style_acc"] + 0.5 * m["drivable_miou"])
            m["select_score"] = score
            print("[val] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
            writer.writerow([epoch, "val", "", f"{m['total']:.5f}", f"{score:.5f}"] + [""] * 10)
            log_f.flush()
            best_loss = min(best_loss, m["total"])
            if score > best_score:
                best_score = score
                torch.save(ckpt, out / "mtl_best.pt")
                print(f"[val] best -> {out/'mtl_best.pt'} score={best_score:.4f}", flush=True)

    (out / "cmd.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "epochs": args.epochs, "batch": args.batch, "limit_train": args.limit_train,
        "seg_params_m": n_seg / 1e6, "best_val_loss": best_loss,
        "best_val_score": best_score,
        "minutes": (time.time() - t0) / 60,
    }, indent=2))
    log_f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
