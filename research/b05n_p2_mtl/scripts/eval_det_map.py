"""Evaluate 10-class mAP50 on the original B05N holdout-1000 split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtl.model import build_model  # noqa: E402

NAMES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike",
         "traffic light", "traffic sign"]


def load_image(path: Path):
    img = Image.open(path).convert("RGB").resize((512, 288), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()[None]


def load_gt(path: Path):
    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            rows.append([float(x) for x in p])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def ap101(tp, fp, n_gt):
    if n_gt == 0:
        return None
    order = np.argsort(-tp[0])
    tp = np.cumsum(tp[1][order])
    fp = np.cumsum(fp[1][order])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    grid = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(grid, mrec, mpre)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/bdd100k")
    ap.add_argument("--names", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--det-ckpt", default="weights/b05n_p2_multiclass.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids = [line.strip() for line in Path(args.names).read_text().splitlines() if line.strip()]
    if args.limit:
        ids = ids[: args.limit]
    model = build_model(args.det_ckpt, device="cuda")
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    records = {c: ([], []) for c in range(10)}  # scores+tp, fp
    n_gt = np.zeros(10, dtype=int)
    img_dir = Path(args.root) / "images" / "flat" / "val"
    gt_dir = Path(args.root) / "label_mtl" / "det" / "val"

    for i, sid in enumerate(ids, 1):
        x = load_image(img_dir / f"{sid}.jpg").cuda()
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            out = model(x, run_det=True)[1]
        y = out[0] if isinstance(out, tuple) else out
        y = y[0].float()
        xywh = y[:4].T
        scores_all = y[4:].T
        xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2,
                          xywh[:, :2] + xywh[:, 2:] / 2], 1)
        xyxy = xyxy * torch.tensor([2.5, 2.5, 2.5, 2.5], device=xyxy.device)
        gt = load_gt(gt_dir / f"{sid}.txt")
        if len(gt):
            gt_xyxy = np.stack([(gt[:, 1] - gt[:, 3] / 2) * 1280,
                                (gt[:, 2] - gt[:, 4] / 2) * 720,
                                (gt[:, 1] + gt[:, 3] / 2) * 1280,
                                (gt[:, 2] + gt[:, 4] / 2) * 720], 1)
            gt_cls = gt[:, 0].astype(int)
            for c in range(10):
                n_gt[c] += int((gt_cls == c).sum())
        else:
            gt_xyxy = np.zeros((0, 4), dtype=np.float32)
            gt_cls = np.zeros(0, dtype=int)

        for c in range(10):
            sc = scores_all[:, c]
            keep = sc > args.conf
            if not bool(keep.any()):
                continue
            boxes = xyxy[keep]
            sc = sc[keep]
            idx = nms(boxes, sc, args.nms_iou)
            boxes, sc = boxes[idx].cpu().numpy(), sc[idx].cpu().numpy()
            g = gt_xyxy[gt_cls == c]
            order = np.argsort(-sc)
            boxes, sc = boxes[order], sc[order]
            matched = np.zeros(len(g), dtype=bool)
            ious = iou_matrix(boxes, g)
            for j in range(len(boxes)):
                tp = 0
                if len(g):
                    cand = np.where(~matched)[0]
                    if len(cand):
                        best = cand[np.argmax(ious[j, cand])]
                        if ious[j, best] >= args.match_iou:
                            tp = 1
                            matched[best] = True
                records[c][0].append(float(sc[j]))
                records[c][1].append(tp)
        if i % 200 == 0:
            print(f"{i}/{len(ids)}", flush=True)

    per_class = {}
    for c in range(10):
        sc = np.asarray(records[c][0])
        tp = np.asarray(records[c][1])
        fp = 1 - tp
        ap50 = ap101((sc, tp), (sc, fp), int(n_gt[c]))
        per_class[NAMES[c]] = {"ap50": ap50, "gt": int(n_gt[c]), "dets": int(len(sc))}
    valid = [v["ap50"] for v in per_class.values() if v["ap50"] is not None]
    result = {
        "checkpoint": args.ckpt,
        "images": len(ids),
        "map50": float(np.mean(valid)) if valid else None,
        "per_class": per_class,
    }
    print(json.dumps(result, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
