"""Run the multi-task model and render detection + lane category/style + drivable overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtl.model import (  # noqa: E402
    DRIVABLE, LANE_CATEGORIES, LANE_STYLES, build_model,
)

DET_NAMES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike",
             "light", "sign"]
LANE_COLORS = [
    (255, 0, 255), (255, 140, 0), (0, 255, 255), (255, 255, 0),
    (0, 220, 0), (150, 100, 255), (255, 255, 255), (255, 200, 0),
]
SCALE = 2
W, H = 512, 288


def font(size=14):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def detect_from_output(det, conf=0.25, iou=0.45):
    y = det[0] if isinstance(det, tuple) else det
    y = y[0]
    boxes = y[:4].T
    scores = y[4:].T
    cls_conf, cls = scores.max(1)
    keep = cls_conf > conf
    boxes, cls_conf, cls = boxes[keep], cls_conf[keep], cls[keep]
    if boxes.numel():
        xyxy = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2], 1)
        idx = nms(xyxy, cls_conf, iou)
        xyxy, cls_conf, cls = xyxy[idx], cls_conf[idx], cls[idx]
    else:
        xyxy = torch.zeros((0, 4), device=boxes.device)
    return xyxy.cpu().numpy(), cls_conf.cpu().numpy(), cls.cpu().numpy()


def draw_boxes(draw, boxes, scores, classes, color=(0, 255, 0)):
    Wc, Hc = W * SCALE, H * SCALE
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        x1 = int(np.clip(x1 * SCALE, 0, Wc - 1)); x2 = int(np.clip(x2 * SCALE, 0, Wc - 1))
        y1 = int(np.clip(y1 * SCALE, 0, Hc - 1)); y2 = int(np.clip(y2 * SCALE, 0, Hc - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{DET_NAMES[int(c)]} {s:.2f}"
        tw = draw.textlength(label, font=font(12))
        ty = max(0, y1 - 14)
        draw.rectangle([x1, ty, min(Wc - 1, x1 + tw + 4), ty + 14], fill=color)
        draw.text((x1 + 2, ty), label, fill=(0, 0, 0), font=font(12))


def overlay(base, mask, colors, alpha=180):
    arr = np.asarray(base).astype(np.float32)
    m = np.asarray(mask)
    for cid, col in colors.items():
        sel = m == cid
        if sel.any():
            arr[sel] = arr[sel] * (1 - alpha / 255) + np.array(col) * (alpha / 255)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def make_panel(img, title, text_color=(255, 255, 255)):
    canvas = Image.new("RGB", (W * SCALE, H * SCALE + 26), (16, 16, 16))
    canvas.paste(img, (0, 26))
    d = ImageDraw.Draw(canvas)
    d.text((8, 5), title, fill=text_color, font=font(15))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/bdd100k")
    ap.add_argument("--ckpt", default="runs/stageA/mtl_last.pt")
    ap.add_argument("--out", default="runs/stageA/viz")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.load(open(Path(args.root) / "label_mtl" / "index_val.json"))
    by_id = {r["id"]: r for r in index}

    # 8 scenes: 4 crosswalk (night/day/dawn + busiest), 2 night lanes, 2 day lanes
    cross = [r for r in index if r["crosswalk_px"] > 800]
    lane = [r for r in index if r["lane"] and r["crosswalk_px"] <= 800 and r["lane_px"] > 2500]
    chosen, used = [], set()

    def add(r):
        if r and r["id"] not in used and len(chosen) < args.num:
            chosen.append(r)
            used.add(r["id"])

    for tod in ("night", "daytime", "dawn/dusk"):
        cand = sorted([r for r in cross if r["timeofday"] == tod],
                      key=lambda r: -r["crosswalk_px"])
        add(cand[0] if cand else None)
    for r in sorted(cross, key=lambda r: -r["crosswalk_px"]):
        if len(chosen) >= 4:
            break
        add(r)

    night = sorted([r for r in lane if r["timeofday"] == "night"], key=lambda r: -r["lane_px"])
    day = sorted([r for r in lane if r["timeofday"] == "daytime"], key=lambda r: -r["lane_px"])
    for group, need in ((night, 2), (day, 2)):
        seen, added = set(), 0
        for r in group:
            if added >= need:
                break
            if r["scene"] not in seen:
                seen.add(r["scene"])
                add(r)
                added += 1
    for r in sorted(lane, key=lambda r: -r["lane_px"]):
        add(r)
    picks = chosen[: args.num]

    model = build_model("weights/b05n_p2_multiclass.pt", device="cuda")
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    report = []
    for i, rec in enumerate(picks):
        sid = rec["id"]
        img = Image.open(Path(args.root) / "images" / "flat" / "val" / f"{sid}.jpg").convert("RGB")
        img = img.resize((W, H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr.transpose(2, 0, 1)).float()[None].cuda()

        with torch.no_grad(), torch.amp.autocast("cuda"):
            seg, det, _ = model(img_t, run_det=True)
        seg = {k: F.interpolate(v, size=(H, W), mode="bilinear", align_corners=False)
               for k, v in seg.items()}
        lane_pred = seg["lane_cat"][0].argmax(0).cpu().numpy()
        style_pred = seg["lane_style"][0].argmax(0).cpu().numpy()
        drv_pred = seg["drivable"][0].argmax(0).cpu().numpy()
        boxes, scores, classes = detect_from_output(det, args.conf)

        base = img.resize((W * SCALE, H * SCALE), Image.NEAREST)

        p1 = base.copy()
        draw_boxes(ImageDraw.Draw(p1), boxes, scores, classes)

        drv_cols = {0: (0, 255, 80), 1: (0, 160, 255)}
        p2 = overlay(base, np.kron(drv_pred, np.ones((SCALE, SCALE), dtype=np.uint8)),
                     drv_cols, alpha=110)
        draw_boxes(ImageDraw.Draw(p2), boxes, scores, classes)

        cat_mask = np.kron(lane_pred, np.ones((SCALE, SCALE), dtype=np.uint8))
        p3 = overlay(base, cat_mask, {c: LANE_COLORS[c] for c in range(8)}, alpha=230)
        draw_boxes(ImageDraw.Draw(p3), boxes, scores, classes)

        style_mask = np.kron(style_pred, np.ones((SCALE, SCALE), dtype=np.uint8))
        lane_hi = np.kron(lane_pred, np.ones((SCALE, SCALE), dtype=np.uint8))
        style_mask[lane_hi == 8] = 255
        p4 = overlay(base, style_mask, {0: (255, 60, 60), 1: (0, 230, 255)}, alpha=240)
        draw_boxes(ImageDraw.Draw(p4), boxes, scores, classes)

        panels = [
            make_panel(p1, "1. detection (B05N P2, frozen) + 10 classes"),
            make_panel(p2, "2. drivable: direct(open) green | alternative blue"),
            make_panel(p3, "3. lane category: crosswalk=magenta | white=cyan | yellow=yellow"),
            make_panel(p4, "4. lane style: solid=red | dashed=cyan"),
        ]
        sheet = Image.new("RGB", (W * SCALE * 2, (H * SCALE + 26) * 2), (0, 0, 0))
        for k, p in enumerate(panels):
            sheet.paste(p, ((k % 2) * W * SCALE, (k // 2) * (H * SCALE + 26)))
        sheet.save(out / f"{i:02d}_{sid}.png")

        n_cross = int(((lane_pred == 0)).sum())
        report.append({"id": sid, "boxes": int(len(boxes)), "crosswalk_px": n_cross,
                       "lane_px": int((lane_pred != 8).sum()),
                       "solid_px": int(((style_pred == 0) & (lane_pred != 8)).sum()),
                       "dashed_px": int(((style_pred == 1) & (lane_pred != 8)).sum()),
                       "direct_px": int((drv_pred == 0).sum()),
                       "alternative_px": int((drv_pred == 1).sum())})
        print(i, sid, report[-1], flush=True)

    json.dump(report, open(out / "pred_report.json", "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
