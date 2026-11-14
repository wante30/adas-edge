"""Compare PyTorch FP32, ONNX FP32, RKNN FP16 and RKNN INT8 on the same frames.

Run on the PC (WSL) after the board has produced rknn_*/<name>.npy outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import (
    CLASS_NAMES,
    draw_legend,
    masks_from_logits,
    overlay_masks,
)


def load_torch_model(repo_root: Path, checkpoint: Path):
    from candidate_plus_torch import load_model

    return load_model(checkpoint)


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def mask_metrics(reference, candidate):
    """reference/candidate: bool [C,H,W] at native resolution."""
    result = {}
    for c, name in enumerate(CLASS_NAMES):
        ref = reference[c]
        cand = candidate[c]
        tp = int(np.logical_and(ref, cand).sum())
        fp = int(np.logical_and(~ref, cand).sum())
        fn = int(np.logical_and(ref, ~cand).sum())
        union = tp + fp + fn
        iou = tp / union if union else None
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
        result[name] = {
            "iou": round(float(iou), 6) if iou is not None else None,
            "f1": round(float(f1), 6) if f1 is not None else None,
            "disagreement_pixels": fp + fn,
            "disagreement_fraction": round(float((fp + fn) / ref.size), 6),
        }
    return result


def run_onnx(session, nchw):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: nchw})[0]


def panel(image, title):
    image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(title, font, scale, thickness)
    x = max(18, image.shape[1] - text_w - 18)
    y = 38
    cv2.putText(
        image,
        title,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        title,
        (x, y),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--torch-checkpoint", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--rknn-fp16-dir", type=Path, required=True)
    parser.add_argument("--rknn-int8-dir", type=Path, required=True)
    parser.add_argument("--onnx-qat", type=Path, help="QAT QDQ ONNX reference")
    parser.add_argument("--rknn-qat-dir", type=Path, help="RKNN built from the QAT QDQ ONNX")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    import onnxruntime as ort
    import torch

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    qat_session = (
        ort.InferenceSession(str(args.onnx_qat), providers=["CPUExecutionProvider"])
        if args.onnx_qat
        else None
    )
    model = load_torch_model(args.repo_root, args.torch_checkpoint)

    per_image = []
    color_check = {"rgb": [], "bgr": []}
    figure_candidates = []

    for row in rows:
        name = row["name"]
        rgb = np.load(row["input"]).astype(np.uint8)[0]
        original = cv2.imread(row["original"], cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(row["original"])
        geometry = row["geometry"]
        nchw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0

        with torch.no_grad():
            torch_logits = model(torch.from_numpy(nchw)).cpu().numpy()
        onnx_logits = run_onnx(session, nchw)
        bgr_nchw = np.ascontiguousarray(rgb[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        onnx_bgr_logits = run_onnx(session, bgr_nchw)

        rknn_fp16 = np.load(args.rknn_fp16_dir / (name + ".npy")).astype(np.float32)
        rknn_int8 = np.load(args.rknn_int8_dir / (name + ".npy")).astype(np.float32)

        fp32_masks, fp32_prob = masks_from_logits(onnx_logits, geometry, args.threshold)
        int8_masks, int8_prob = masks_from_logits(rknn_int8, geometry, args.threshold)
        fp16_masks, fp16_prob = masks_from_logits(rknn_fp16, geometry, args.threshold)
        qat_logits = run_onnx(qat_session, nchw) if qat_session is not None else None
        qat_masks = qat_prob = None
        if qat_logits is not None:
            qat_masks, qat_prob = masks_from_logits(qat_logits, geometry, args.threshold)
        rknn_qat = None
        rknn_qat_masks = rknn_qat_prob = None
        if args.rknn_qat_dir is not None:
            rknn_qat = np.load(args.rknn_qat_dir / (name + ".npy")).astype(np.float32)
            rknn_qat_masks, rknn_qat_prob = masks_from_logits(
                rknn_qat, geometry, args.threshold
            )

        torch_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
        fp16_mae = float(np.mean(np.abs(rknn_fp16 - onnx_logits)))
        int8_mae = float(np.mean(np.abs(rknn_int8 - onnx_logits)))
        color_check["rgb"].append(float(np.mean(np.abs(rknn_fp16 - onnx_logits))))
        color_check["bgr"].append(float(np.mean(np.abs(rknn_fp16 - onnx_bgr_logits))))

        row_result = {
            "name": name,
            "source": row.get("source"),
            "original": row["original"],
            "geometry": geometry,
            "torch_vs_onnx": {
                "logits_max_abs": round(torch_diff, 8),
                "logits_mae": round(float(np.mean(np.abs(torch_logits - onnx_logits))), 8),
            },
            "rknn_fp16_vs_onnx": {
                "logits_mae": round(fp16_mae, 8),
                "logits_max_abs": round(float(np.max(np.abs(rknn_fp16 - onnx_logits))), 8),
                "prob_mae": round(float(np.mean(np.abs(fp16_prob - fp32_prob))), 8),
                "masks": mask_metrics(fp32_masks, fp16_masks),
            },
            "rknn_int8_vs_onnx": {
                "logits_mae": round(int8_mae, 8),
                "logits_max_abs": round(float(np.max(np.abs(rknn_int8 - onnx_logits))), 8),
                "prob_mae": round(float(np.mean(np.abs(int8_prob - fp32_prob))), 8),
                "masks": mask_metrics(fp32_masks, int8_masks),
            },
        }
        if qat_logits is not None:
            row_result["qat_onnx_vs_fp32"] = {
                "logits_mae": round(float(np.mean(np.abs(qat_logits - onnx_logits))), 8),
                "logits_max_abs": round(float(np.max(np.abs(qat_logits - onnx_logits))), 8),
                "prob_mae": round(float(np.mean(np.abs(qat_prob - fp32_prob))), 8),
                "masks": mask_metrics(fp32_masks, qat_masks),
            }
        if rknn_qat is not None and qat_logits is not None:
            row_result["rknn_qat_vs_qat_onnx"] = {
                "logits_mae": round(float(np.mean(np.abs(rknn_qat - qat_logits))), 8),
                "logits_max_abs": round(float(np.max(np.abs(rknn_qat - qat_logits))), 8),
                "prob_mae": round(float(np.mean(np.abs(rknn_qat_prob - qat_prob))), 8),
                "masks": mask_metrics(qat_masks, rknn_qat_masks),
            }
            row_result["rknn_qat_vs_fp32"] = {
                "logits_mae": round(float(np.mean(np.abs(rknn_qat - onnx_logits))), 8),
                "logits_max_abs": round(float(np.max(np.abs(rknn_qat - onnx_logits))), 8),
                "prob_mae": round(float(np.mean(np.abs(rknn_qat_prob - fp32_prob))), 8),
                "masks": mask_metrics(fp32_masks, rknn_qat_masks),
            }
        per_image.append(row_result)

        # Save a same-frame comparison for the first frame and a frame with
        # substantial foreground; four panels at half resolution.
        foreground = int(fp32_masks[0].sum() + fp32_masks[1].sum() + fp32_masks[2].sum())
        selected_masks = rknn_qat_masks if rknn_qat_masks is not None else int8_masks
        figure_candidates.append(
            (foreground, name, original, fp32_masks, selected_masks, rknn_qat_masks is not None)
        )

    # Summary across images.
    summary = {}
    metric_keys = ["rknn_fp16_vs_onnx", "rknn_int8_vs_onnx"]
    if qat_session is not None:
        metric_keys.append("qat_onnx_vs_fp32")
    if args.rknn_qat_dir is not None:
        metric_keys.extend(["rknn_qat_vs_qat_onnx", "rknn_qat_vs_fp32"])
    for key in metric_keys:
        summary[key] = {
            "logits_mae": summarize([r[key]["logits_mae"] for r in per_image]),
            "logits_max_abs": summarize([r[key]["logits_max_abs"] for r in per_image]),
            "prob_mae": summarize([r[key]["prob_mae"] for r in per_image]),
            "classes": {},
        }
        for cname in CLASS_NAMES:
            summary[key]["classes"][cname] = {
                "iou": summarize([r[key]["masks"][cname]["iou"] for r in per_image]),
                "f1": summarize([r[key]["masks"][cname]["f1"] for r in per_image]),
                "disagreement_fraction": summarize(
                    [r[key]["masks"][cname]["disagreement_fraction"] for r in per_image]
                ),
            }
    summary["torch_vs_onnx"] = {
        "logits_max_abs": summarize([r["torch_vs_onnx"]["logits_max_abs"] for r in per_image]),
        "logits_mae": summarize([r["torch_vs_onnx"]["logits_mae"] for r in per_image]),
    }
    summary["color_order_check"] = {
        "rknn_fp16_vs_onnx_rgb_mae": summarize(color_check["rgb"]),
        "rknn_fp16_vs_onnx_bgr_mae": summarize(color_check["bgr"]),
        "winner": "RGB" if np.mean(color_check["rgb"]) <= np.mean(color_check["bgr"]) else "BGR",
    }
    summary["samples"] = len(per_image)

    result = {"onnx": str(args.onnx), "torch_checkpoint": str(args.torch_checkpoint), "summary": summary, "per_image": per_image}
    (args.out_dir / "accuracy.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (args.out_dir / "accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "torch_onnx_max_abs",
                "fp16_logits_mae",
                "int8_logits_mae",
                "fp16_prob_mae",
                "int8_prob_mae",
            ]
            + ["fp32_iou_" + c for c in CLASS_NAMES]
            + ["fp16_iou_" + c for c in CLASS_NAMES]
            + ["int8_iou_" + c for c in CLASS_NAMES]
        )
        for row in per_image:
            writer.writerow(
                [
                    row["name"],
                    row["torch_vs_onnx"]["logits_max_abs"],
                    row["rknn_fp16_vs_onnx"]["logits_mae"],
                    row["rknn_int8_vs_onnx"]["logits_mae"],
                    row["rknn_fp16_vs_onnx"]["prob_mae"],
                    row["rknn_int8_vs_onnx"]["prob_mae"],
                ]
                + ["" for _ in CLASS_NAMES]
                + [row["rknn_fp16_vs_onnx"]["masks"][c]["iou"] for c in CLASS_NAMES]
                + [row["rknn_int8_vs_onnx"]["masks"][c]["iou"] for c in CLASS_NAMES]
            )

    # Pick the frame with the largest total foreground for the same-frame figure.
    figure_candidates.sort(key=lambda item: item[0], reverse=True)
    for rank, (_, name, original, fp32_masks, int8_masks, using_qat) in enumerate(
        figure_candidates[:3]
    ):
        h, w = original.shape[:2]
        scale = min(1.0, 960.0 / w)
        size = (int(round(w * scale)), int(round(h * scale)))

        def shrink(image):
            return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        fp32_overlay = draw_legend(overlay_masks(original.copy(), fp32_masks, 0.45))
        int8_overlay = draw_legend(overlay_masks(original.copy(), int8_masks, 0.45))
        diff = np.zeros_like(original)
        union = np.zeros(original.shape[:2], dtype=bool)
        for c in range(4):
            change = np.logical_xor(fp32_masks[c], int8_masks[c])
            union |= change
            diff[change] = (0, 255, 255)
        int8_title = "RK3566 INT8 QAT" if using_qat else "RK3566 INT8"
        top = np.hstack(
            [panel(shrink(original), "original"), panel(shrink(fp32_overlay), "FP32 ONNX")]
        )
        bottom = np.hstack(
            [panel(shrink(int8_overlay), int8_title), panel(shrink(diff), "FP32 / INT8 disagreement")]
        )
        canvas = np.vstack([top, bottom])
        cv2.imwrite(
            str(figure_dir / ("same_frame_%s.jpg" % name)),
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if rank == 0:
            cv2.imwrite(
                str(figure_dir / "same_frame_fp32_vs_rk3566_int8.jpg"),
                canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
