"""Far traffic-light evaluation on the 50-image BDD50 refined GT set."""

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


def load_gt(gt_dir: Path):
    gt = {}
    for path in sorted(gt_dir.glob("*.json")):
        data = json.loads(path.read_text())
        boxes = []
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                if obj.get("category") != "traffic light":
                    continue
                b = obj["box2d"]
                x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
        gt[path.stem] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return gt


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def ap101(scores, tp, n_gt):
    if n_gt == 0:
        return None
    order = np.argsort(-scores)
    tp_sorted = tp[order]
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(1 - tp_sorted)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(np.mean(np.interp(np.linspace(0, 1, 101), mrec, mpre)))


def match(pred, gt, iou_thr=0.5):
    order = np.argsort(-pred[:, 4])
    pred = pred[order]
    matched = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(pred), dtype=np.float32)
    ious = iou_matrix(pred[:, :4], gt)
    for i in range(len(pred)):
        if len(gt) == 0:
            break
        cand = np.where(~matched)[0]
        if not len(cand):
            break
        best = cand[np.argmax(ious[i, cand])]
        if ious[i, best] >= iou_thr:
            tp[i] = 1
            matched[best] = True
    return order, tp, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/bdd100k")
    ap.add_argument("--gt-dir", default="eval_sets/bdd50_gt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--roi-top", type=float, default=0.60)
    ap.add_argument("--recall-conf", type=float, default=0.25)
    args = ap.parse_args()

    gt = load_gt(Path(args.gt_dir))
    model = build_model("weights/b05n_p2_multiclass.pt", device="cuda")
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    full_scores, full_tp = [], []
    roi_scores, roi_tp = [], []
    n_full = n_roi = 0
    tiny = {"gt": 0, "hit": 0}
    img_dir = Path(args.root) / "images" / "flat" / "val"
    roi_h = args.roi_top * 720

    for sid, g in gt.items():
        img = Image.open(img_dir / f"{sid}.jpg").convert("RGB").resize((512, 288), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1)).float()[None].cuda()
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            out = model(x, run_det=True)[1]
        y = (out[0] if isinstance(out, tuple) else out)[0].float()
        sc = y[4 + 8]
        keep = sc > args.conf
        boxes = y[:4].T[keep]
        sc = sc[keep]
        if len(boxes):
            xyxy = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                              boxes[:, :2] + boxes[:, 2:] / 2], 1) * 2.5
            idx = nms(xyxy, sc, args.nms_iou)
            xyxy, sc = xyxy[idx].cpu().numpy(), sc[idx].cpu().numpy()
            pred = np.concatenate([xyxy, sc[:, None]], 1)
        else:
            pred = np.zeros((0, 5), dtype=np.float32)

        n_full += len(g)
        order, tp, _ = match(pred, g)
        full_scores.extend(pred[order, 4].tolist())
        full_tp.extend(tp.tolist())

        roi_pred = pred[(pred[:, 1] + pred[:, 3]) / 2 < roi_h] if len(pred) else pred
        roi_gt = g[(g[:, 1] + g[:, 3]) / 2 < roi_h] if len(g) else g
        n_roi += len(roi_gt)
        order, tp, _ = match(roi_pred, roi_gt)
        roi_scores.extend(roi_pred[order, 4].tolist())
        roi_tp.extend(tp.tolist())

        # small GT 8-16px, recall at operational confidence
        widths = g[:, 2] - g[:, 0]
        small = g[(widths >= 8) & (widths <= 16)]
        tiny["gt"] += len(small)
        if len(small):
            op = pred[pred[:, 4] >= args.recall_conf]
            _, _, matched = match(op, small)
            tiny["hit"] += int(matched.sum())

    res = {
        "checkpoint": args.ckpt,
        "images": len(gt),
        "gt_full": n_full,
        "gt_roi": n_roi,
        "ap50_full": ap101(np.asarray(full_scores), np.asarray(full_tp), n_full),
        "ap50_roi": ap101(np.asarray(roi_scores), np.asarray(roi_tp), n_roi),
        "recall_small_8_16_at_conf": tiny["hit"] / max(tiny["gt"], 1),
        "small_gt": tiny["gt"],
        "small_hit": tiny["hit"],
    }
    print(json.dumps(res, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
