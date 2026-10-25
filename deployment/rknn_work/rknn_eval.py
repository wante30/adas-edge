"""Compare ONNX vs RKNN simulator: box parity and multiclass mAP50."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05k_common import box_iou, decode_yolo11, evaluate_predictions  # noqa: E402
from tools.b05n_tl_eval import decode_class  # noqa: E402

CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"]


def to_linux(path: str) -> Path:
    if path.startswith("/mnt/"):
        return Path(path)
    drive, rest = os.path.splitdrive(path)
    if drive:
        return Path("/mnt") / drive[0].lower() / rest.lstrip("\\/").replace("\\", "/")
    return Path(path)


def prepare_full(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return rgb, tensor


def prepare_spec(bgr: np.ndarray, roi_top: float = 0.60) -> tuple[np.ndarray, np.ndarray]:
    roi = bgr[: int(round(bgr.shape[0] * roi_top))]
    resized = cv2.resize(roi, (512, 192), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return rgb, tensor


def decode_multiclass(raw: Any, conf: float, iou: float) -> dict[str, list[dict]]:
    out = {}
    for class_id, name in enumerate(CLASSES):
        items = decode_class(raw, class_id, conf, iou)
        for item in items:
            item["bbox"] = [item["bbox"][0] * 2.5, item["bbox"][1] * 2.5,
                            item["bbox"][2] * 2.5, item["bbox"][3] * 2.5]
            item["class_name"] = name
        out[name] = items
    return out


def match_stats(a: list[dict], b: list[dict], iou_thr: float = 0.5) -> dict[str, Any]:
    used: set[int] = set()
    ious, confs = [], []
    for det in sorted(a, key=lambda x: -x["confidence"]):
        best, best_iou = None, 0.0
        for index, ref in enumerate(b):
            if index in used:
                continue
            overlap = box_iou(det["bbox"], ref["bbox"])
            if overlap > best_iou:
                best_iou, best = overlap, index
        if best is not None and best_iou >= iou_thr:
            used.add(best)
            ious.append(best_iou)
            confs.append(abs(det["confidence"] - b[best]["confidence"]))
    return {"a": len(a), "b": len(b), "matched": len(ious),
            "unmatched_a": len(a) - len(ious), "unmatched_b": len(b) - len(ious),
            "mean_iou": float(np.mean(ious)) if ious else None,
            "mean_conf_delta": float(np.mean(confs)) if confs else None}


def load_bdd50_gt(images_dir: Path, gt_dir: Path, roi_top: float) -> dict[str, list[dict]]:
    ground_truth: dict[str, list[dict]] = {}
    for gt_path in sorted(gt_dir.glob("*.json")):
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        name = str(data["name"])
        image = cv2.imread(str(images_dir / f"{name}.jpg"))
        if image is None:
            continue
        height, width = image.shape[:2]
        roi_height = int(round(height * roi_top))
        targets = []
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                if obj.get("category") != "traffic light":
                    continue
                box = obj.get("box2d", {})
                x1, y1, x2, y2 = (float(box.get(k, 0)) for k in ("x1", "y1", "x2", "y2"))
                if y2 <= roi_height and x2 > x1 and y2 > y1:
                    targets.append({"bbox": [x1, y1, x2, y2], "width_px": x2 - x1})
        ground_truth[name] = targets
    return ground_truth


def load_gt(labels: Path, name: str, width: int, height: int, class_id: int) -> list[dict]:
    path = labels / f"{Path(name).stem}.txt"
    targets = []
    if not path.exists():
        return targets
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != class_id:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:])
        targets.append({"bbox": [(cx - w / 2) * width, (cy - h / 2) * height,
                                 (cx + w / 2) * width, (cy + h / 2) * height],
                        "width_px": w * width})
    return targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("multiclass", "single"), required=True)
    ap.add_argument("--rknn", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--split-onnx", type=Path, required=True)
    ap.add_argument("--precision", choices=("fp16", "int8"), default="int8")
    ap.add_argument("--calib-dir", type=Path)
    ap.add_argument("--algorithm", default="normal", choices=("normal", "mmse", "kl_divergence"))
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--gt-dir", type=Path)
    ap.add_argument("--offset", type=int, default=200)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--mode", choices=("parity", "map", "both", "tlmap", "diag"), default="parity")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from rknn.api import RKNN
    import onnxruntime as ort

    names = sorted(args.images.glob("*.jpg"))[args.offset: args.offset + args.limit]
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3566",
                quantized_method="channel", quantized_algorithm=args.algorithm)
    assert rknn.load_onnx(model=str(args.split_onnx)) == 0
    dataset = None
    if args.precision == "int8":
        if args.calib_dir is None:
            raise SystemExit("--calib-dir is required for int8")
        calib = sorted(to_linux(str(args.calib_dir)).glob("*.jpg"))
        listing = args.output.parent / f"calib_{args.output.stem}.txt"
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text("\n".join(str(p) for p in calib) + "\n", encoding="utf-8")
        dataset = str(listing)
    assert rknn.build(do_quantization=args.precision == "int8", dataset=dataset) == 0
    assert rknn.init_runtime() == 0
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    parity: dict[str, dict] = {}
    raw_diffs = []
    times = []
    gt_by_class: dict[str, dict[str, list[dict]]] = {name: {} for name in CLASSES}
    gt_single: dict[str, list[dict]] = {}
    preds_single_rknn: dict[str, list[dict]] = {}
    preds_single_onnx: dict[str, list[dict]] = {}
    if args.mode == "tlmap":
        if not args.gt_dir:
            raise SystemExit("--gt-dir is required for tlmap")
        gt_single = load_bdd50_gt(args.images, args.gt_dir, 0.60)
    preds_rknn: dict[str, dict[str, list[dict]]] = {name: {} for name in CLASSES}
    preds_onnx: dict[str, dict[str, list[dict]]] = {name: {} for name in CLASSES}

    for source in names:
        bgr = cv2.imread(str(source))
        rgb, tensor = prepare_spec(bgr) if args.kind == "single" else prepare_full(bgr)
        started = time.perf_counter()
        outs = rknn.inference(inputs=[rgb])
        boxes, scores = outs[0], outs[1]
        if boxes.ndim == 3:
            boxes = boxes[0]
        if scores.ndim == 3:
            scores = scores[0]
        probs = 1.0 / (1.0 + np.exp(-scores.astype(np.float64)))
        rk = np.concatenate([boxes.astype(np.float32), probs.astype(np.float32)], axis=0)[None]
        times.append((time.perf_counter() - started) * 1000.0)
        on = session.run(None, {input_name: tensor})[0]
        if rk.shape != on.shape:
            rk = rk.reshape(on.shape)
        raw_diffs.append(float(np.abs(rk.astype(np.float32) - on).mean()))

        if args.kind == "single":
            boxes_rk = decode_yolo11(rk, conf=args.conf, iou=args.iou)
            boxes_on = decode_yolo11(on, conf=args.conf, iou=args.iou)
            stats = match_stats(boxes_rk, boxes_on)
            if args.mode == "tlmap":
                for det in boxes_rk:
                    det["bbox"] = [det["bbox"][0] / 0.4, det["bbox"][1] / (192.0 / (720 * 0.6)),
                                   det["bbox"][2] / 0.4, det["bbox"][3] / (192.0 / (720 * 0.6))]
                for det in boxes_on:
                    det["bbox"] = [det["bbox"][0] / 0.4, det["bbox"][1] / (192.0 / (720 * 0.6)),
                                   det["bbox"][2] / 0.4, det["bbox"][3] / (192.0 / (720 * 0.6))]
                preds_single_rknn[source.name] = boxes_rk
                preds_single_onnx[source.name] = boxes_on
            key = "traffic light"
            parity.setdefault(key, {"a": 0, "b": 0, "matched": 0, "unmatched_a": 0, "unmatched_b": 0,
                                    "ious": [], "confs": []})
            entry = parity[key]
            for field in ("a", "b", "matched", "unmatched_a", "unmatched_b"):
                entry[field] += stats[field]
            if stats["mean_iou"] is not None:
                entry["ious"].append(stats["mean_iou"])
            if stats["mean_conf_delta"] is not None:
                entry["confs"].append(stats["mean_conf_delta"])
        else:
            rk_boxes = decode_multiclass(rk, args.conf, args.iou)
            on_boxes = decode_multiclass(on, args.conf, args.iou)
            for name in CLASSES:
                stats = match_stats(rk_boxes[name], on_boxes[name])
                entry = parity.setdefault(name, {"a": 0, "b": 0, "matched": 0, "unmatched_a": 0,
                                                 "unmatched_b": 0, "ious": [], "confs": []})
                for field in ("a", "b", "matched", "unmatched_a", "unmatched_b"):
                    entry[field] += stats[field]
                if stats["mean_iou"] is not None:
                    entry["ious"].append(stats["mean_iou"])
                if stats["mean_conf_delta"] is not None:
                    entry["confs"].append(stats["mean_conf_delta"])
                if args.mode in ("map", "both", "diag"):
                    preds_rknn[name][source.name] = rk_boxes[name]
                    preds_onnx[name][source.name] = on_boxes[name]
            if args.mode in ("map", "both", "diag"):
                for name in CLASSES:
                    gt_by_class[name][source.name] = load_gt(args.labels, source.name, 1280, 720,
                                                             CLASSES.index(name))

    result: dict[str, Any] = {
        "kind": args.kind, "rknn": str(args.rknn), "onnx": str(args.onnx),
        "images": len(names), "thresholds": {"conf": args.conf, "nms_iou": args.iou},
        "raw_mean_abs_diff": float(np.mean(raw_diffs)),
        "runtime_ms": {"mean": statistics.fmean(times), "p50": float(np.percentile(times, 50)),
                       "p95": float(np.percentile(times, 95))},
        "parity": {},
    }
    for name, entry in parity.items():
        result["parity"][name] = {
            "rknn_boxes": entry["a"], "onnx_boxes": entry["b"], "matched": entry["matched"],
            "unmatched_rknn": entry["unmatched_a"], "unmatched_onnx": entry["unmatched_b"],
            "mean_iou": float(np.mean(entry["ious"])) if entry["ious"] else None,
            "mean_conf_delta": float(np.mean(entry["confs"])) if entry["confs"] else None,
        }
    if args.mode == "diag" and args.kind == "multiclass":
        diag = {}
        for name in CLASSES:
            gt = gt_by_class[name]
            if sum(len(v) for v in gt.values()) == 0:
                continue
            row = {}
            for tag, table in (("rknn", preds_rknn), ("onnx", preds_onnx)):
                tp = fp = fn = 0
                for image_key, targets in gt.items():
                    preds = [p for p in table[name].get(image_key, []) if p["confidence"] >= 0.25]
                    used: set[int] = set()
                    for det in sorted(preds, key=lambda x: -x["confidence"]):
                        best, best_iou = None, 0.0
                        for idx, target in enumerate(targets):
                            if idx in used:
                                continue
                            overlap = box_iou(det["bbox"], target["bbox"])
                            if overlap > best_iou:
                                best_iou, best = overlap, idx
                        if best is not None and best_iou >= 0.5:
                            used.add(best)
                            tp += 1
                        else:
                            fp += 1
                    fn += len(targets) - len(used)
                row[tag] = {"tp": tp, "fp": fp, "fn": fn,
                            "precision": tp / (tp + fp) if tp + fp else 0.0,
                            "recall": tp / (tp + fn) if tp + fn else 0.0}
            diag[name] = row
        result["diag_conf_0.25"] = diag
    if args.mode == "tlmap" and args.kind == "single":
        result["tl_map"] = {
            tag: evaluate_predictions(gt_single, table, conf=0.25, iou=0.5, ap_floor=0.001)
            for tag, table in (("rknn", preds_single_rknn), ("onnx", preds_single_onnx))
        }
    if args.mode in ("map", "both") and args.kind == "multiclass":
        aps = {"rknn": {}, "onnx": {}}
        for name in CLASSES:
            total_gt = sum(len(v) for v in gt_by_class[name].values())
            if total_gt == 0:
                continue
            for tag, table in (("rknn", preds_rknn), ("onnx", preds_onnx)):
                aps[tag][name] = evaluate_predictions(gt_by_class[name], table[name], conf=0.25,
                                                      iou=0.5, ap_floor=0.001)["ap50"]
        result["map50"] = aps
        result["map50_mean"] = {tag: float(np.mean(list(v.values()))) if v else None for tag, v in aps.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()
