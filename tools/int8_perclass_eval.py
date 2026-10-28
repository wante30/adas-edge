"""Per-class INT8 quality check on real RK3566 outputs (not just IoU/matching).

Compares a PyTorch FP32 detection dump against a decoded RKNN INT8 dump over
the same images, on the frozen holdout GT, and reports per-class AP50 and
recall@0.25 for both backends plus the delta. AP50 uses pycocotools over the
competition class subset; recall uses the same greedy one-to-one IoU-0.5
operating point as every other report in this project.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tools.coco_eval_helpers import (
    CLASSES,
    build_ground_truth,
    operating_point,
    read_bdd_rows,
)
from tools.compare_detection_dumps import board_dir_to_dump, load_dump

# COCO-80 model index -> competition class name. Everything else is ignored.
MODEL_INDEX_TO_CLASS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}
CATEGORY_IDS = {name: index + 1 for index, name in enumerate(CLASSES)}


def dump_to_coco(dump: dict) -> list[dict]:
    detections = []
    for index, entry in enumerate(dump["per_image"]):
        for row in entry["detections"]:
            class_id, score, x1, y1, x2, y2 = row
            name = MODEL_INDEX_TO_CLASS.get(int(class_id))
            if name is None:
                continue
            detections.append(
                {
                    "image_id": index + 1,
                    "category_id": CATEGORY_IDS[name],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return detections


def per_class_ap50(coco_gt: dict, detections: list[dict]) -> dict[str, float]:
    coco = COCO()
    coco.dataset = json.loads(json.dumps(coco_gt))
    coco.createIndex()
    coco_dt = coco.loadRes(detections) if detections else coco.loadRes([])
    evaluator = COCOeval(coco, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"]
    results = {}
    for index, name in enumerate(CLASSES):
        values = precision[:, :, index, 0, -1]
        values = values[values > -1]
        results[name] = float(values.mean()) if values.size else 0.0
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="FP32 detection dump")
    parser.add_argument("--board-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--label", default="INT8-perclass")
    parser.add_argument("--board-conf", type=float, default=0.001, help="decode floor for the board dump")
    parser.add_argument("--board-iou", type=float, default=0.7)
    parser.add_argument("--recall-conf", type=float, default=0.25)
    args = parser.parse_args()

    fp32 = load_dump(args.reference)
    int8 = board_dir_to_dump(args.board_dir, args.manifest, args.board_conf, args.board_iou)
    image_names = [entry["image"] for entry in fp32["per_image"]]
    if image_names != [entry["image"] for entry in int8["per_image"]]:
        # order both by image name so the GT indices line up
        order = {name: index + 1 for index, name in enumerate(image_names)}
        fp32["per_image"].sort(key=lambda entry: order[entry["image"]])
        int8_names = {entry["image"]: entry for entry in int8["per_image"]}
        int8["per_image"] = [int8_names[name] for name in image_names]

    rows = read_bdd_rows(args.bdd_root / "bdd100k.csv")
    coco_gt, _gt_stats = build_ground_truth(rows, image_names, width=1280, height=720)

    fp32_dets = dump_to_coco(fp32)
    int8_dets = dump_to_coco(int8)
    fp32_ap = per_class_ap50(coco_gt, fp32_dets)
    int8_ap = per_class_ap50(coco_gt, int8_dets)

    fp32_point = operating_point(fp32_dets, coco_gt, args.recall_conf)["classes"]
    int8_point = operating_point(int8_dets, coco_gt, args.recall_conf)["classes"]

    report = {
        "label": args.label,
        "images": len(image_names),
        "board_conf_floor": args.board_conf,
        "recall_conf": args.recall_conf,
        "classes": {},
    }
    for name in CLASSES:
        report["classes"][name] = {
            "AP50_fp32": fp32_ap[name],
            "AP50_int8": int8_ap[name],
            "recall_fp32": fp32_point[name]["recall"],
            "recall_int8": int8_point[name]["recall"],
            "precision_fp32": fp32_point[name]["precision"],
            "precision_int8": int8_point[name]["precision"],
        }
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    lines = [
        f"# {args.label}",
        "",
        f"图像数 {len(image_names)}，recall 口径 conf={args.recall_conf}（贪心一对一 IoU0.5），"
        f"AP50 口径 floor={args.board_conf}（pycocotools）。",
        "",
        "| 类别 | AP50 FP32 | AP50 INT8 | Δ | recall FP32 | recall INT8 | Δ | precision FP32 | precision INT8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CLASSES:
        c = report["classes"][name]
        lines.append(
            f"| {name} | {c['AP50_fp32']:.4f} | {c['AP50_int8']:.4f} | {c['AP50_int8'] - c['AP50_fp32']:+.4f} "
            f"| {c['recall_fp32']:.4f} | {c['recall_int8']:.4f} | {c['recall_int8'] - c['recall_fp32']:+.4f} "
            f"| {c['precision_fp32']:.4f} | {c['precision_int8']:.4f} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
