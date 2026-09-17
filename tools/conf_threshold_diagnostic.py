"""Confidence-threshold diagnostic for a dumped prediction set.

Answers one question per class: is the low recall an absence of predictions, or
are low-scoring predictions being filtered by the operating threshold?

For every ground-truth box it finds the highest-scoring prediction of the same
class that reaches the IoU threshold. The resulting score distribution is the
"reachable" recall curve, and comparing it against the recall at a given
confidence separates the two failure modes:

* **no prediction** - the box never gets a match, even at the score floor;
* **filtered out** - the box gets a match, but its best score is below the
  operating confidence.

The tool only reads a prediction dump, so a threshold sweep never re-runs the
model and every confidence level is compared on exactly the same predictions.
It produces evidence; it does not change any deployment configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # allow `python tools/conf_threshold_diagnostic.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.coco_eval_helpers import (
    CATEGORY_IDS,
    CLASSES,
    build_ground_truth,
    iou,
    operating_point,
    read_bdd_rows,
)

DEFAULT_CONFS = (0.05, 0.10, 0.15, 0.20, 0.25)
SCORE_BANDS = ((0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 1.01))


def load_predictions(path: Path) -> tuple[list[dict], dict[int, str], list[str]]:
    """Read a prediction dump into (per-image rows, model names, image names)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_names = {int(key): value for key, value in payload["model_names"].items()}
    per_image = [(entry["image"], [tuple(row) for row in entry["detections"]]) for entry in payload["per_image"]]
    image_names = [name for name, _ in per_image]
    return per_image, model_names, image_names


def best_score_per_ground_truth(
    detections: list[dict],
    coco_gt: dict,
    iou_threshold: float = 0.5,
) -> dict[str, dict]:
    """Per class, the best matching prediction score for each ground-truth box.

    ``None`` means no prediction of that class reached ``iou_threshold`` at all,
    even at the dump's score floor - the box is unreachable, not merely filtered.
    """
    name_by_id = {value: key for key, value in CATEGORY_IDS.items()}
    gt_by_key: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for annotation in coco_gt["annotations"]:
        x, y, w, h = annotation["bbox"]
        gt_by_key[(annotation["image_id"], annotation["category_id"])].append([x, y, x + w, y + h])
    pred_by_key: dict[tuple[int, int], list[tuple[float, list[float]]]] = defaultdict(list)
    for detection in detections:
        x, y, w, h = detection["bbox"]
        pred_by_key[(detection["image_id"], detection["category_id"])].append(
            (float(detection["score"]), [x, y, x + w, y + h])
        )

    result: dict[str, dict] = {
        name: {"gt": 0, "reachable": 0, "band_counts": {f"{low:.2f}-{high:.2f}": 0 for low, high in SCORE_BANDS}, "scores": []}
        for name in CLASSES
    }
    for key, boxes in gt_by_key.items():
        name = name_by_id.get(key[1])
        if name is None:
            continue
        predictions = pred_by_key.get(key, [])
        for gt_box in boxes:
            entry = result[name]
            entry["gt"] += 1
            best = 0.0
            for score, pred_box in predictions:
                if score > best and iou(pred_box, gt_box) >= iou_threshold:
                    best = score
            if best <= 0.0:
                continue
            entry["reachable"] += 1
            entry["scores"].append(best)
            for low, high in SCORE_BANDS:
                if low <= best < high:
                    entry["band_counts"][f"{low:.2f}-{high:.2f}"] += 1
                    break
    for name, entry in result.items():
        entry["unreachable"] = entry["gt"] - entry["reachable"]
        entry["reachable_ratio"] = entry["reachable"] / entry["gt"] if entry["gt"] else 0.0
        entry["scores"] = sorted(entry["scores"])
    return result


def sweep(
    detections: list[dict],
    coco_gt: dict,
    confs: tuple[float, ...] = DEFAULT_CONFS,
    iou_threshold: float = 0.5,
) -> dict[str, dict]:
    """Operating point at each confidence level."""
    return {f"{conf:.2f}": operating_point(detections, coco_gt, conf=conf, iou_threshold=iou_threshold) for conf in confs}


def classify_failure(entry: dict) -> str:
    """Turn one class's reachability into the Phase 5 verdict."""
    if entry["gt"] == 0:
        return "no-ground-truth"
    if entry["reachable"] == 0:
        return "A: no prediction reaches the box even at the score floor"
    above_operating = sum(
        count for band, count in entry["band_counts"].items() if float(band.split("-")[1]) > 0.25
    )
    if above_operating == 0:
        return "B: predictions exist but every one is below 0.25"
    if entry["reachable_ratio"] < 0.5:
        return "A+B: most boxes are unreachable, the rest are near the threshold"
    return "mostly reachable; threshold is not the bottleneck"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--confs", default=",".join(f"{conf:g}" for conf in DEFAULT_CONFS))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--label")
    args = parser.parse_args()

    confs = tuple(float(part) for part in args.confs.split(",") if part.strip())
    per_image, model_names, image_names = load_predictions(args.predictions)
    image_names = image_names[: args.limit]
    rows = read_bdd_rows(args.bdd_root / "bdd100k.csv")
    coco_gt, _ = build_ground_truth(rows, image_names, width=1280, height=720)

    # Rebuild COCO detections from the dump, restricted to the same image slice.
    from tools.coco_eval_helpers import predictions_to_coco

    detections, _stats = predictions_to_coco(per_image, image_names, model_names, score_floor=0.0)

    sweeps = sweep(detections, coco_gt, confs)
    reachable = best_score_per_ground_truth(detections, coco_gt)
    verdicts = {name: classify_failure(entry) for name, entry in reachable.items()}

    report = {
        "label": args.label or args.predictions.stem,
        "predictions": str(args.predictions),
        "images": len(image_names),
        "iou_match": 0.5,
        "confs": list(confs),
        "sweep": sweeps,
        "reachability": {
            name: {key: value for key, value in entry.items() if key != "scores"} for name, entry in reachable.items()
        },
        "verdict": verdicts,
        "note": "conf sweep reuses one prediction dump; the model is never re-run, so every conf sees identical predictions",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Confidence diagnostic: {report['label']}",
        "",
        f"- {report['images']} holdout images, IoU >= 0.5, predictions from `{args.predictions.name}`",
        "",
        "## Reachability at the score floor (is there any prediction at all?)",
        "",
        "| class | GT | reachable | unreachable | reachable ratio | verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in CLASSES:
        entry = reachable[name]
        lines.append(
            f"| {name} | {entry['gt']} | {entry['reachable']} | {entry['unreachable']} "
            f"| {entry['reachable_ratio']:.3f} | {verdicts[name]} |"
        )
    lines += ["", "## Best matching score bands", "", "| class | " + " | ".join(band for band in reachable[CLASSES[0]]["band_counts"]) + " |", "| --- | " + " | ".join("---:" for _ in SCORE_BANDS) + " |"]
    for name in CLASSES:
        counts = reachable[name]["band_counts"]
        lines.append(f"| {name} | " + " | ".join(str(counts[band]) for band in counts) + " |")
    lines += ["", "## Recall and F1 versus confidence", "", "| class | metric | " + " | ".join(f"{conf:.2f}" for conf in confs) + " |", "| --- | --- | " + " | ".join("---:" for _ in confs) + " |"]
    for name in CLASSES:
        for metric in ("recall", "f1"):
            cells = [f"{sweeps[f'{conf:.2f}']['classes'][name][metric]:.3f}" for conf in confs]
            lines.append(f"| {name} | {metric} | " + " | ".join(cells) + " |")
    for metric in ("macro_recall", "macro_f1"):
        cells = [f"{sweeps[f'{conf:.2f}'][metric]:.3f}" for conf in confs]
        lines.append(f"| **macro** | {metric} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
