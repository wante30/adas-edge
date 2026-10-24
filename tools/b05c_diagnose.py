"""B-05C offline traffic-light diagnosis on the B-03D 50-image holdout.

Consumes the board prediction dump (same images, BDD KL and Specialist) and
the BDD color JSON.  Reports detection AP/recall by size, BDD/Specialist/fusion
matches, and colour confusion on GT vs predicted boxes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from competition_ai.traffic_light import TrafficLightState, classify_traffic_light
from competition_ai.traffic_light_tracking import classify_lamp_color


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def load_gt(json_dir):
    gt = {}
    for path in sorted(Path(json_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                if obj.get("category") != "traffic light":
                    continue
                box = obj.get("box2d") or {}
                try:
                    bbox = [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
                except (KeyError, TypeError, ValueError):
                    continue
                color = str((obj.get("attributes") or {}).get("trafficLightColor", "missing")).lower()
                rows.append({"bbox": bbox, "color": color})
        gt[path.stem + ".jpg"] = rows
    return gt


def ap50(by_image, gt):
    # AP must rank every prediction globally by confidence.  A per-image sort
    # changes the precision/recall envelope and is not COCO-style AP50.
    records = []
    for name, preds in by_image.items():
        for pred in preds:
            records.append((float(pred["confidence"]), name, pred["bbox"]))
    records.sort(key=lambda item: item[0], reverse=True)
    used = {name: set() for name in gt}
    total_gt = sum(len(rows) for rows in gt.values())
    if not records or total_gt == 0:
        return {"ap50": 0.0, "tp": 0, "fp": 0, "fn": total_gt}
    tp = fp = 0
    points = []
    for _, name, bbox in records:
        gts = gt.get(name, [])
        best, index = 0.0, None
        for i, gt_box in enumerate(gts):
            if i in used[name]:
                continue
            overlap = iou(bbox, gt_box["bbox"])
            if overlap > best:
                best, index = overlap, i
        hit = index is not None and best >= 0.5
        if hit:
            used[name].add(index)
            tp += 1
        else:
            fp += 1
        points.append((tp / total_gt, tp / max(1, tp + fp)))
    # 101-point interpolated AP, matching the B-03D style.
    aps = []
    for step in range(101):
        recall = step / 100.0
        precision = max((p for r, p in points if r >= recall), default=0.0)
        aps.append(precision)
    return {"ap50": sum(aps) / len(aps), "tp": tp, "fp": fp, "fn": total_gt - tp}


def at_conf(by_image, gt, conf):
    tp = fp = fn = 0
    for name, preds in by_image.items():
        gts = gt.get(name, [])
        used = set()
        for pred in sorted((p for p in preds if p["confidence"] >= conf),
                           key=lambda item: item["confidence"], reverse=True):
            best, index = 0.0, None
            for i, gt_box in enumerate(gts):
                if i in used:
                    continue
                overlap = iou(pred["bbox"], gt_box["bbox"])
                if overlap > best:
                    best, index = overlap, i
            if index is not None and best >= 0.5:
                used.add(index)
                tp += 1
            else:
                fp += 1
        fn += len(gts) - len(used)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"conf": conf, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-9, precision + recall)}


def size_recall(by_image, gt, conf=0.25):
    bins = [(0, 8), (8, 16), (16, 32), (32, 10**9)]
    result = {}
    for low, high in bins:
        total = hit = 0
        for name, gts in gt.items():
            preds = [p for p in by_image.get(name, []) if p["confidence"] >= conf]
            used = set()
            for gt_box in gts:
                width = gt_box["bbox"][2] - gt_box["bbox"][0]
                if not (low <= width < high):
                    continue
                total += 1
                for i, pred in enumerate(preds):
                    if i in used:
                        continue
                    if iou(pred["bbox"], gt_box["bbox"]) >= 0.5:
                        used.add(i)
                        hit += 1
                        break
        result[f"{low}-{high if high < 10**9 else 'inf'}px"] = {
            "gt": total, "tp": hit, "recall": hit / max(1, total),
        }
    return result


def fuse(bdd, specialist):
    out = {}
    for name in sorted(set(bdd) | set(specialist)):
        candidates = []
        for source, rows in (("bdd", bdd.get(name, [])), ("specialist", specialist.get(name, []))):
            for row in rows:
                candidates.append((float(row["confidence"]), source, row["bbox"]))
        candidates.sort(reverse=True)
        kept = []
        for conf, source, box in candidates:
            if any(iou(box, old["bbox"]) >= 0.5 for old in kept):
                continue
            kept.append({"confidence": conf, "source": source, "bbox": box})
        out[name] = kept
    return out


def color_name(frame, box):
    state = classify_traffic_light(frame, {"bbox": box, "confidence": 1.0}, min_confidence=0.0)
    return state.value


def collect_color(gt, images, preds, source):
    matrix = defaultdict(Counter)
    rows = []
    for name, gts in gt.items():
        image = cv2.imread(str(Path(images) / name))
        if image is None:
            continue
        for gt_box in gts:
            gt_color = gt_box["color"]
            if gt_color not in ("red", "green", "yellow"):
                continue
            crop = image[max(0, int(gt_box["bbox"][1])):int(gt_box["bbox"][3]),
                         max(0, int(gt_box["bbox"][0])):int(gt_box["bbox"][2])]
            if crop.size == 0:
                continue
            gt_pred = color_name(image, gt_box["bbox"])
            best, best_iou = None, 0.0
            for pred in preds.get(name, []):
                overlap = iou(pred["bbox"], gt_box["bbox"])
                if overlap > best_iou:
                    best, best_iou = pred, overlap
            pred_color = "NO_DET"
            if best is not None and best_iou >= 0.5:
                pred_color = color_name(image, best["bbox"])
            matrix[gt_color][pred_color] += 1
            rows.append({"image": name, "gt": gt_color, "gt_box_color": gt_pred,
                         "pred_box_color": pred_color, "bbox_iou": best_iou,
                         "crop_w": crop.shape[1], "crop_h": crop.shape[0]})
    return matrix, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--images", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--out", default="outputs/b05c_summary.json")
    args = parser.parse_args()
    data = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    bdd, specialist = data["bdd"], data["specialist"]
    gt = load_gt(args.gt_json)
    merged = fuse(bdd, specialist)
    summary = {
        "gt_traffic_lights": sum(len(v) for v in gt.values()),
        "bdd": {"ap": ap50(bdd, gt), "at_conf": [at_conf(bdd, gt, c) for c in (0.25, 0.35, 0.5)],
                "size": size_recall(bdd, gt)},
        "specialist": {"ap": ap50(specialist, gt),
                       "at_conf": [at_conf(specialist, gt, c) for c in (0.25, 0.35, 0.5)],
                       "size": size_recall(specialist, gt)},
        "fusion": {"ap": ap50(merged, gt),
                   "at_conf": [at_conf(merged, gt, c) for c in (0.25, 0.35, 0.5)],
                   "size": size_recall(merged, gt)},
    }
    for key, source in (("bdd", bdd), ("specialist", specialist)):
        matrix, rows = collect_color(gt, args.images, source, source)
        summary[key]["color"] = {gt_name: dict(counter) for gt_name, counter in matrix.items()}
        summary[key]["color_rows"] = rows
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k]["at_conf"] for k in ("bdd", "specialist", "fusion")}, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()



