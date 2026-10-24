"""B-05L BDD50 failure visualizations and cross-model TP differences.

Draws GT in green, matched predictions in cyan, unmatched predictions in red and
missed GT in orange.  Contact sheets are for human review; they do not turn a
box into ground truth and do not claim video-level recall.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_ai.traffic_light_tracking import classify_lamp_color  # noqa: E402
from tools.b05l_eval import iou, match_image  # noqa: E402

COLOR_BGR = {"RED": (40, 40, 235), "GREEN": (40, 210, 60), "YELLOW": (30, 220, 240), "UNKNOWN": (220, 220, 220)}


def draw_scene(image: np.ndarray, gt: list[dict[str, Any]], predictions: list[dict[str, Any]], *, conf: float, iou_threshold: float) -> tuple[np.ndarray, dict[str, Any]]:
    canvas = image.copy()
    pairs, tp, fp, fn = match_image(gt, predictions, conf=conf, iou_threshold=iou_threshold)
    matched_gt = {index for _, index in pairs}
    matched_pred = {id(prediction) for prediction, _ in pairs}
    for index, target in enumerate(gt):
        x1, y1, x2, y2 = (int(round(value)) for value in target["bbox"])
        color = (40, 200, 40) if index in matched_gt else (30, 140, 245)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"GT {target.get('color_gt', 'none')}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    for prediction in predictions:
        if float(prediction["confidence"]) < conf:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in prediction["bbox"])
        matched = id(prediction) in matched_pred
        color = (220, 200, 40) if matched else COLOR_BGR["RED"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{prediction.get('color', '?')} {float(prediction['confidence']):.2f}"
        cv2.putText(canvas, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (18, 24, 28), -1)
    cv2.putText(canvas, f"GT {len(gt)}  TP {tp}  FP {fp}  FN {fn}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 245, 248), 1, cv2.LINE_AA)
    return canvas, {"tp": tp, "fp": fp, "fn": fn, "pairs": len(pairs)}


def contact_sheet(items: list[tuple[np.ndarray, str]], path: Path, *, columns: int = 2, width: int = 640) -> None:
    if not items:
        return
    height = int(width * 9 / 16)
    rows = (len(items) + columns - 1) // columns
    sheet = np.full((rows * (height + 26), columns * (width + 8), 3), 28, np.uint8)
    for index, (image, caption) in enumerate(items):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, columns)
        y, x = row * (height + 26), col * (width + 8)
        sheet[y:y + height, x:x + width] = image
        cv2.putText(sheet, caption[:100], (x + 4, y + height + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (235, 240, 245), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cases_for_model(name: str, data: dict[str, Any], images_dir: Path, gt: dict[str, list[dict[str, Any]]], out: Path, limit: int = 8) -> dict[str, Any]:
    predictions = data["predictions_by_image"]
    fp_rows = []
    fn_rows = []
    unknown_rows = []
    multiple_rows = []
    for image_name, targets in gt.items():
        image = cv2.imread(str(images_dir / f"{image_name}.jpg"))
        if image is None:
            continue
        pairs, tp, fp, fn = match_image(targets, predictions.get(image_name, []), conf=0.25, iou_threshold=0.5)
        matched = {index for _, index in pairs}
        for prediction in predictions.get(image_name, []):
            if float(prediction["confidence"]) < 0.25:
                continue
            if not any(prediction is item[0] for item in pairs):
                fp_rows.append((float(prediction["confidence"]), image_name, prediction["bbox"], image))
        for index, target in enumerate(targets):
            if index not in matched:
                fn_rows.append((float(target["width_px"]), image_name, target["bbox"], target.get("color_gt"), image))
        for prediction, index in pairs:
            observation = classify_lamp_color(image, prediction["bbox"])
            if not observation.valid:
                unknown_rows.append((float(prediction["confidence"]), image_name, prediction["bbox"], image))
        if tp >= 3:
            multiple_rows.append((tp, image_name, image))
    selections = {
        "false_positives": sorted(fp_rows, key=lambda row: -row[0])[:limit],
        "missed_small_gt": sorted(fn_rows, key=lambda row: row[0])[:limit],
        "color_unknown": sorted(unknown_rows, key=lambda row: -row[0])[:limit],
        "multiple_lights": sorted(multiple_rows, key=lambda row: -row[0])[:limit],
    }
    records = {}
    for category, rows in selections.items():
        records[category] = []
        sheet_items = []
        for row in rows:
            if category == "false_positives":
                confidence, image_name, bbox, image = row
                scene, _ = draw_scene(image, gt[image_name], predictions.get(image_name, []), conf=0.25, iou_threshold=0.5)
                caption = f"{image_name} FP conf={confidence:.2f}"
                records[category].append({"image": image_name, "bbox": bbox, "confidence": confidence})
            elif category == "missed_small_gt":
                width_px, image_name, bbox, color, image = row
                scene, _ = draw_scene(image, gt[image_name], predictions.get(image_name, []), conf=0.25, iou_threshold=0.5)
                caption = f"{image_name} missed GT {width_px:.1f}px {color}"
                records[category].append({"image": image_name, "bbox": bbox, "width_px": width_px, "color_gt": color})
            elif category == "color_unknown":
                confidence, image_name, bbox, image = row
                scene, _ = draw_scene(image, gt[image_name], predictions.get(image_name, []), conf=0.25, iou_threshold=0.5)
                caption = f"{image_name} UNKNOWN conf={confidence:.2f}"
                records[category].append({"image": image_name, "bbox": bbox, "confidence": confidence})
            else:
                tp, image_name, image = row
                scene, _ = draw_scene(image, gt[image_name], predictions.get(image_name, []), conf=0.25, iou_threshold=0.5)
                caption = f"{image_name} TP={tp}"
                records[category].append({"image": image_name, "tp": tp})
            sheet_items.append((scene, caption))
        contact_sheet(sheet_items, out / f"{name}_{category}.jpg")
        if rows:
            first = rows[0]
            image = first[-1]
            scene, _ = draw_scene(image, gt[first[1]], predictions.get(first[1], []), conf=0.25, iou_threshold=0.5)
            cv2.imwrite(str(out / f"{name}_{category}_keyframe.jpg"), scene, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return records


def cross_model(name_a: str, data_a: dict[str, Any], name_b: str, data_b: dict[str, Any], images_dir: Path, gt: dict[str, list[dict[str, Any]]], out: Path, limit: int = 6) -> dict[str, Any]:
    rows = []
    for image_name, targets in gt.items():
        image = cv2.imread(str(images_dir / f"{image_name}.jpg"))
        if image is None:
            continue
        pairs_a, tp_a, _, _ = match_image(targets, data_a["predictions_by_image"].get(image_name, []), conf=0.25, iou_threshold=0.5)
        pairs_b, tp_b, _, _ = match_image(targets, data_b["predictions_by_image"].get(image_name, []), conf=0.25, iou_threshold=0.5)
        if tp_a == tp_b:
            continue
        rows.append((abs(tp_a - tp_b), image_name, tp_a, tp_b, image))
    rows.sort(key=lambda row: -row[0])
    selections = {
        "baseline_more": [row for row in rows if row[2] > row[3]][:limit],
        "candidate_more": [row for row in rows if row[3] > row[2]][:limit],
    }
    records = {}
    for category, items in selections.items():
        records[category] = []
        sheet = []
        for _, image_name, tp_a, tp_b, image in items:
            scene, _ = draw_scene(image, gt[image_name], data_b["predictions_by_image"].get(image_name, []), conf=0.25, iou_threshold=0.5)
            caption = f"{image_name} {name_a} TP={tp_a} | {name_b} TP={tp_b}"
            sheet.append((scene, caption))
            records[category].append({"image": image_name, name_a: tp_a, name_b: tp_b})
        contact_sheet(sheet, out / f"compare_{name_a}_vs_{name_b}_{category}.jpg")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True, help="LABEL=path/to/eval.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    models = {}
    for value in args.eval:
        label, path = value.split("=", 1)
        models[label] = load_eval(Path(path))
    first = next(iter(models.values()))
    images_dir = Path(first["dataset"]["images_dir"])
    gt_dir = Path(first["dataset"]["gt_dir"])
    from tools.b05l_eval import load_gt
    _, gt, _ = load_gt(images_dir, gt_dir, 0.60)
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, data in models.items():
        records[name] = cases_for_model(name, data, images_dir, gt, args.output, args.limit)
    comparisons = {}
    if "yolo11n" in models:
        for name, data in models.items():
            if name == "yolo11n":
                continue
            comparisons[name] = cross_model("yolo11n", models["yolo11n"], name, data, images_dir, gt, args.output, args.limit)
    (args.output / "selection.json").write_text(json.dumps({"models": records, "comparisons": comparisons}, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "models": list(models), "records": {k: {c: len(v) for c, v in value.items()} for k, value in records.items()}}, indent=2))


if __name__ == "__main__":
    main()
