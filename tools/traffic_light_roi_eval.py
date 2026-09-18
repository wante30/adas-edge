"""Traffic-light ROI feasibility study (no training).

Crops a region of interest from the full 1280x720 road frame, letterboxes the
crop onto a static canvas, runs the original COCO-80 yolo11n.pt and maps boxes
back to full-frame coordinates. Only the ``traffic light`` class is evaluated,
against the frozen BDD holdout GT, so the numbers answer one question: can a
front-only upscaled ROI branch make traffic lights usable without training?

Metrics per (ROI, canvas, conf): AP50 (pycocotools, catIds=[traffic light]),
greedy one-to-one precision/recall at IoU 0.5 (same convention as the
historical operating point - NOT mAP), and candidate coverage: the fraction of
TL ground-truth boxes that have at least one candidate detection at
IoU >= 0.5 with any confidence above the floor. "GT with no candidate" is the
complement and is the hard physical limit on recall.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.coco_eval_helpers import (
    CLASSES,
    build_ground_truth,
    operating_point,
    read_bdd_rows,
)
from tools.profile_rknn_single_stream import letterbox_bgr

TL_CLASS = "traffic light"
TL_CATEGORY_ID = CLASSES.index(TL_CLASS) + 1  # COCO ids are 1-based
FLOOR = 0.001

# (label, x0, y0, x1, y1) in fractional coordinates of the full frame.
# gt95 comes from the frozen holdout TL-GT statistics: y0 5th pct = 0.065,
# y1 95th pct = 0.585, x1 95th pct = 0.846 -> full width x top 60% covers
# roughly 95% of boxes; the script measures the real coverage per ROI.
ROI_DEFINITIONS: dict[str, tuple[float, float, float, float]] = {
    "ROI-A-full": (0.0, 0.0, 1.0, 1.0),
    "ROI-B-top60": (0.0, 0.0, 1.0, 0.60),
    "ROI-C-top50": (0.0, 0.0, 1.0, 0.50),
    "ROI-D-gt95": (0.0, 0.0, 0.95, 0.60),
}


def parse_canvases(text: str) -> list[tuple[int, int]]:
    canvases = []
    for token in text.split(","):
        width, _, height = token.strip().lower().partition("x")
        canvases.append((int(width), int(height)))
    return canvases


def roi_pixel_box(roi: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    return (
        round(x0 * width),
        round(y0 * height),
        round(x1 * width),
        round(y1 * height),
    )


def gt_coverage(annotations: list[dict], roi: tuple[float, float, float, float], width: int, height: int) -> float:
    """Fraction of TL GT boxes fully inside the ROI, in full-frame pixels."""
    px = roi_pixel_box(roi, width, height)
    inside = 0
    total = 0
    for annotation in annotations:
        if annotation["category_id"] != TL_CATEGORY_ID:
            continue
        x, y, w, h = annotation["bbox"]
        total += 1
        if x >= px[0] and y >= px[1] and x + w <= px[2] and y + h <= px[3]:
            inside += 1
    return inside / total if total else 0.0


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--label", default="TL-ROI")
    parser.add_argument("--canvases", default="512x288,640x352")
    parser.add_argument(
        "--rois",
        default="",
        help="comma list of ROI labels to run (default: all ROI_DEFINITIONS)",
    )
    parser.add_argument("--confs", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = parser.parse_args()

    width_px, height_px = 1280, 720
    canvases = parse_canvases(args.canvases)
    confs = [float(c) for c in args.confs.split(",")]
    roi_filter = {token.strip() for token in args.rois.split(",") if token.strip()}
    roi_items = [(label, roi) for label, roi in ROI_DEFINITIONS.items() if not roi_filter or label in roi_filter]
    if roi_filter:
        unknown = roi_filter - set(ROI_DEFINITIONS)
        if unknown:
            raise SystemExit(f"unknown ROI labels: {sorted(unknown)}")

    rows = read_bdd_rows(args.bdd_root / "bdd100k.csv")
    names_file = Path("D:/Datasets/adas_edge_training/bdd_holdout_1000_names.txt")
    image_names = [line.strip() for line in names_file.read_text().splitlines() if line.strip()]
    if args.limit:
        image_names = image_names[: args.limit]
    coco_gt, gt_stats = build_ground_truth(rows, image_names, width=width_px, height=height_px)

    tl_annotations = [a for a in coco_gt["annotations"] if a["category_id"] == TL_CATEGORY_ID]
    tl_gt_total = len(tl_annotations)
    images_root = args.bdd_root / "images" / "100k" / "val"

    from ultralytics import YOLO

    model = YOLO(args.model)
    tl_model_index = [int(k) for k, v in model.names.items() if v == TL_CLASS]
    if len(tl_model_index) != 1:
        raise SystemExit(f"model does not expose exactly one {TL_CLASS!r} class: {model.names}")
    tl_model_index = tl_model_index[0]

    results: list[dict] = []
    started = time.perf_counter()
    for canvas_index, (canvas_w, canvas_h) in enumerate(canvases):
        canvas_label = f"{canvas_w}x{canvas_h}"
        for roi_label, roi in roi_items:
            coverage = gt_coverage(coco_gt["annotations"], roi, width_px, height_px)
            per_image: list[dict] = []
            forward_ms: list[float] = []
            import cv2

            for image_index, name in enumerate(image_names):
                image = cv2.imread(str(images_root / name))
                if image is None:
                    raise RuntimeError(f"cannot read {name}")
                rx0, ry0, rx1, ry1 = roi_pixel_box(roi, width_px, height_px)
                crop = image[ry0:ry1, rx0:rx1]
                canvas, scale, pad_x, pad_y = letterbox_bgr(crop, (canvas_w, canvas_h))
                t0 = time.perf_counter()
                prediction = model.predict(
                    source=canvas,
                    imgsz=(canvas_h, canvas_w),
                    conf=FLOOR,
                    max_det=300,
                    device=args.device,
                    rect=False,
                    verbose=False,
                )[0]
                forward_ms.append((time.perf_counter() - t0) * 1000)
                for box, cls, score in zip(
                    prediction.boxes.xyxy.cpu().tolist(),
                    prediction.boxes.cls.cpu().tolist(),
                    prediction.boxes.conf.cpu().tolist(),
                ):
                    if int(cls) != tl_model_index:
                        continue
                    # canvas coords -> ROI-crop coords -> full-frame coords
                    fx0 = (box[0] - pad_x) / scale + rx0
                    fy0 = (box[1] - pad_y) / scale + ry0
                    fx1 = (box[2] - pad_x) / scale + rx0
                    fy1 = (box[3] - pad_y) / scale + ry0
                    per_image.append(
                        {
                            "image_id": image_index + 1,
                            "category_id": TL_CATEGORY_ID,
                            "bbox": [fx0, fy0, fx1 - fx0, fy1 - fy0],
                            "score": float(score),
                        }
                    )
                if (image_index + 1) % 200 == 0:
                    print(
                        f"  {canvas_label} {roi_label}: {image_index + 1}/{len(image_names)} images,"
                        f" {len(per_image)} dets so far",
                        flush=True,
                    )

            # AP50 restricted to the traffic-light category
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            coco = COCO()
            coco.dataset = json.loads(json.dumps(coco_gt))
            coco.createIndex()
            coco_dt = coco.loadRes(per_image) if per_image else coco.loadRes([])
            evaluator = COCOeval(coco, coco_dt, iouType="bbox")
            evaluator.params.catIds = [TL_CATEGORY_ID]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            ap50 = float(evaluator.stats[1])

            point_metrics: dict[str, dict] = {}
            for conf in confs:
                point = operating_point(per_image, coco_gt, conf)
                tl = point["classes"][TL_CLASS]
                point_metrics[f"{conf:.2f}"] = {
                    "precision": tl["precision"],
                    "recall": tl["recall"],
                    "tp": tl["tp"],
                    "fp": tl["fp"],
                    "fn": tl["fn"],
                }

            # candidate coverage: TL GT matched by any detection >= FLOOR at IoU 0.5
            gt_boxes: dict[int, list[list[float]]] = {}
            for annotation in tl_annotations:
                x, y, w, h = annotation["bbox"]
                gt_boxes.setdefault(annotation["image_id"], []).append([x, y, x + w, y + h])
            from tools.coco_eval_helpers import iou as iou_fn

            no_candidate = 0
            for image_id, boxes in gt_boxes.items():
                dets = [d for d in per_image if d["image_id"] == image_id]
                hit = [False] * len(boxes)
                for det in sorted(dets, key=lambda d: -d["score"]):
                    best, best_iou = -1, 0.0
                    for gi, gt_box in enumerate(boxes):
                        value = iou_fn(
                            [det["bbox"][0], det["bbox"][1], det["bbox"][0] + det["bbox"][2], det["bbox"][1] + det["bbox"][3]],
                            gt_box,
                        )
                        if value > best_iou:
                            best_iou, best = value, gi
                    if best >= 0 and best_iou >= 0.5:
                        hit[best] = True
                no_candidate += hit.count(False)

            record = {
                "canvas": canvas_label,
                "canvas_w": canvas_w,
                "canvas_h": canvas_h,
                "roi": roi_label,
                "roi_frac": list(roi),
                "roi_px_full_frame": list(roi_pixel_box(roi, width_px, height_px)),
                "roi_area_fraction": (roi[2] - roi[0]) * (roi[3] - roi[1]),
                "tl_gt_coverage": coverage,
                "tl_gt_total": tl_gt_total,
                "tl_gt_no_candidate": no_candidate,
                "candidate_coverage": 1.0 - no_candidate / tl_gt_total if tl_gt_total else 0.0,
                "AP50": ap50,
                "forward_ms_mean": sum(forward_ms) / len(forward_ms) if forward_ms else 0.0,
                "detections_total": len(per_image),
                "points": point_metrics,
            }
            results.append(record)
            print(
                f"== {canvas_label} {roi_label}: AP50={ap50:.4f} coverage={record['candidate_coverage']:.3f}"
                f" R@0.10={point_metrics['0.10']['recall']:.4f} P@0.10={point_metrics['0.10']['precision']:.4f}",
                flush=True,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {
                        "label": args.label,
                        "model": args.model,
                        "images": len(image_names),
                        "tl_gt_total": tl_gt_total,
                        "confs": confs,
                        "floor": FLOOR,
                        "elapsed_s": time.perf_counter() - started,
                        "results": results,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )

    args.out.write_text(
        json.dumps(
            {
                "label": args.label,
                "model": args.model,
                "images": len(image_names),
                "tl_gt_total": tl_gt_total,
                "confs": confs,
                "floor": FLOOR,
                "elapsed_s": time.perf_counter() - started,
                "results": results,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    run()
