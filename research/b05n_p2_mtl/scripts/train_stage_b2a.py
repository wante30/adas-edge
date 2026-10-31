"""B-2a: 3-epoch joint fine-tune of P2/P3 shared features + detection head + seg decoder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtl.data import BDDMTLDataset, collate  # noqa: E402
from mtl.losses import (  # noqa: E402
    batch_stats, drivable_loss, lane_category_loss, lane_style_loss,
)
from mtl.model import build_model  # noqa: E402

UNFREEZE = [0, 1, 2, 3, 4, 5, 6, 12, 15, 18, 19, 20, 21]


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
        "seg": total.detach(), "lane_ce": lc_ce.detach(), "lane_dice": lc_dice.detach(),
        "lane_fg": lc_fg.detach(), "lane_tv": lc_tv.detach(),
        "style_ce": ls_ce.detach(), "drv_ce": dr_ce.detach(),
    }
    return total, parts


def det_batch(batch):
    idxs, clss, bxs = [], [], []
    for i, x in enumerate(batch["boxes"]):
        x = x.cuda(non_blocking=True)
        if x.numel():
            idxs.append(torch.full((x.shape[0],), i, device="cuda"))
            clss.append(x[:, 0])
            bxs.append(x[:, 1:5])
    if not idxs:
        return {"batch_idx": torch.zeros(0, device="cuda"),
                "cls": torch.zeros(0, device="cuda"),
                "bboxes": torch.zeros((0, 4), device="cuda")}
    return {"batch_idx": torch.cat(idxs).long(), "cls": torch.cat(clss).long(),
            "bboxes": torch.cat(bxs)}


@torch.no_grad()
def evaluate(model, loader, lane_weights):
    model.eval()
    agg, n = {}, 0
    for batch in loader:
        batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.amp.autocast("cuda"):
            seg, _, _ = model(batch["image"], run_det=False)
            _, parts = seg_losses(seg, batch, lane_weights)
            stats = batch_stats(seg, batch)
        bs = batch["image"].shape[0]
        n += bs
        for k, v in {**parts, **stats}.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bs
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/bdd100k")
    ap.add_argument("--ckpt", default="weights/b05n_p2_multiclass.pt")
    ap.add_argument("--resume", default="runs/stageB1/mtl_best.pt")
    ap.add_argument("--out", default="runs/stageB2a")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-shared", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=5e-5)
    ap.add_argument("--lr-seg", type=float, default=3e-4)
    ap.add_argument("--det-weight", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(0)
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

    model = build_model(args.ckpt, device="cuda")
    ck = torch.load(args.resume, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"])
    model.set_trainable(UNFREEZE, train_head=True)
    model.det.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, label_smoothing=0.0)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"resumed {args.resume} | trainable {n_train/1e6:.3f}M | unfreeze {UNFREEZE}", flush=True)

    from ultralytics.utils.loss import v8DetectionLoss
    criterion = v8DetectionLoss(model.det)

    head_params = list(model.det.model[-1].parameters())
    head_ids = {id(p) for p in head_params}
    shared_params = [p for p in model.det.parameters() if p.requires_grad and id(p) not in head_ids]
    seg_params = list(model.seg_parameters())
    opt = torch.optim.AdamW([
        {"params": shared_params, "lr": args.lr_shared},
        {"params": head_params, "lr": args.lr_head},
        {"params": seg_params, "lr": args.lr_seg},
    ], weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    total_steps = max(args.epochs * steps_per_epoch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_shared, args.lr_head, args.lr_seg],
        total_steps=total_steps, pct_start=0.05, div_factor=10, final_div_factor=50)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    lane_weights = torch.tensor([25.] * 8 + [1.0], device="cuda")

    log_path = out / "train_log.csv"
    log_f = open(log_path, "a", newline="")
    writer = csv.writer(log_f)
    if log_path.stat().st_size == 0:
        writer.writerow(["epoch", "step", "lr_shared", "loss", "seg", "det",
                         "box", "cls", "dfl", "lane_ce", "lane_fg", "style_ce", "drv_ce",
                         "lane_fg_iou", "crosswalk_iou", "style_acc", "drivable_miou"])

    best_score, best_loss = -1.0, float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, rs, n_seen = 0.0, {}, 0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                seg, det_out, _ = model(batch["image"], run_det=True)
                seg_total, parts = seg_losses(seg, batch, lane_weights)
                dloss, ditems = criterion(det_out, det_batch(batch))
                det_total = dloss.sum() / batch["image"].shape[0]
                total = seg_total + args.det_weight * det_total

            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            bs = batch["image"].shape[0]
            n_seen += bs
            run += float(total) * bs
            rs["seg"] = rs.get("seg", 0.0) + float(seg_total) * bs
            rs["det"] = rs.get("det", 0.0) + float(det_total) * bs
            for k, v in ditems.items():
                rs[k] = rs.get(k, 0.0) + float(v) * bs
            for k, v in parts.items():
                if k == "seg":
                    continue
                rs[k] = rs.get(k, 0.0) + float(v) * bs

            if step % 20 == 0:
                with torch.no_grad():
                    st = batch_stats(seg, batch)
                lr = sched.get_last_lr()
                print(f"e{epoch} {step}/{steps_per_epoch} loss={run/n_seen:.4f} "
                      f"seg={rs['seg']/n_seen:.3f} det={rs['det']/n_seen:.3f} "
                      f"box={rs['box_loss']/n_seen:.3f} cls={rs['cls_loss']/n_seen:.3f} "
                      f"dfl={rs['dfl_loss']/n_seen:.3f} lane_fg_iou={st['lane_fg_iou']:.3f} "
                      f"cw={st['crosswalk_iou']:.3f} style={st['style_acc']:.3f} "
                      f"drv={st['drivable_miou']:.3f} lr={lr[0]:.1e}/{lr[2]:.1e}", flush=True)
                writer.writerow([epoch, step, f"{lr[0]:.3g}", f"{run/n_seen:.5f}",
                                 f"{rs['seg']/n_seen:.5f}", f"{rs['det']/n_seen:.5f}",
                                 f"{rs['box_loss']/n_seen:.5f}", f"{rs['cls_loss']/n_seen:.5f}",
                                 f"{rs['dfl_loss']/n_seen:.5f}", f"{rs['lane_ce']/n_seen:.5f}",
                                 f"{rs['lane_fg']/n_seen:.5f}", f"{rs['style_ce']/n_seen:.5f}",
                                 f"{rs['drv_ce']/n_seen:.5f}", f"{st['lane_fg_iou']:.5f}",
                                 f"{st['crosswalk_iou']:.5f}", f"{st['style_acc']:.5f}",
                                 f"{st['drivable_miou']:.5f}"])
                log_f.flush()

        ckpt = {"model": model.state_dict(),
                "arch": {"mid": 64, "head_ch": 64, "tap_idx": model.tap_idx},
                "source": args.ckpt, "resume": args.resume, "epoch": epoch, "args": vars(args)}
        torch.save(ckpt, out / "mtl_last.pt")
        print(f"[epoch {epoch}] saved elapsed={(time.time()-t0)/60:.1f}min", flush=True)

        if epoch % args.val_every == 0 and len(val_loader):
            m = evaluate(model, val_loader, lane_weights)
            score = (m["lane_fg_iou"] + m["crosswalk_iou"]
                     + 0.5 * m["style_acc"] + 0.5 * m["drivable_miou"])
            m["select_score"] = score
            print("[val] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
            writer.writerow([epoch, "val", "", f"{m['seg']:.5f}", "", "", "", "", "", "", "", "", "",
                             f"{m['lane_fg_iou']:.5f}", f"{m['crosswalk_iou']:.5f}",
                             f"{m['style_acc']:.5f}", f"{m['drivable_miou']:.5f}"])
            log_f.flush()
            best_loss = min(best_loss, m["seg"])
            if score > best_score:
                best_score = score
                torch.save(ckpt, out / "mtl_best.pt")
                print(f"[val] best score={best_score:.4f}", flush=True)

    (out / "cmd.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "epochs": args.epochs, "batch": args.batch, "limit_train": args.limit_train,
        "trainable_m": n_train / 1e6, "unfreeze": UNFREEZE,
        "best_val_score": best_score, "best_seg_loss": best_loss,
        "minutes": (time.time() - t0) / 60,
    }, indent=2))
    log_f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
