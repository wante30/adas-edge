"""Build/validate the multi-task RKNN and report detection + segmentation agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def decode_raw(outputs, h=288, w=512):
    boxes_all, scores_all = [], []
    reg_max = None
    for s in range(len(outputs) // 3):
        box = np.asarray(outputs[s * 3], dtype=np.float32)
        if box.ndim == 3:
            box = box[None]
        cur = box.shape[1] // 4
        reg_max = cur if reg_max is None else reg_max
        hh, ww = box.shape[2:]
        logits = box.reshape(box.shape[0], 4, reg_max, hh * ww)
        logits = logits - logits.max(axis=2, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=2, keepdims=True)
        bins = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)
        d = (p * bins).sum(axis=2)
        gx = np.tile(np.arange(ww, dtype=np.float32), hh)
        gy = np.repeat(np.arange(hh, dtype=np.float32), ww)
        sx, sy = w / ww, h / hh
        left = (gx[None] + 0.5 - d[:, 0]) * sx
        top = (gy[None] + 0.5 - d[:, 1]) * sy
        right = (gx[None] + 0.5 + d[:, 2]) * sx
        bottom = (gy[None] + 0.5 + d[:, 3]) * sy
        boxes_all.append(np.stack(((left + right) / 2, (top + bottom) / 2,
                                   right - left, bottom - top), 1))
        sc = np.asarray(outputs[s * 3 + 1], dtype=np.float32)
        if sc.ndim == 3:
            sc = sc[None]
        scores_all.append(sc.reshape(sc.shape[0], sc.shape[1], -1))
    return np.concatenate((np.concatenate(boxes_all, 2), np.concatenate(scores_all, 2)), 1)


def nms(boxes, scores, thr=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        aa = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        bb = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(aa + bb - inter, 1e-9)
        order = rest[iou <= thr]
    return keep


def dets_from_decoded(det, conf=0.25, nms_iou=0.5):
    xywh = det[:4].T
    scores = det[4:].T
    out = []
    for c in range(scores.shape[1]):
        s = scores[:, c]
        idx = np.flatnonzero(s >= conf)
        if not len(idx):
            continue
        b = xywh[idx]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        keep = nms(xyxy, s[idx], nms_iou)
        for k in keep:
            out.append([*xyxy[k], float(s[idx][k]), c])
    return np.asarray(out, dtype=np.float32).reshape(-1, 6)


def iou_one(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    aa = (box[2] - box[0]) * (box[3] - box[1])
    bb = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(aa + bb - inter, 1e-9)


def match_dets(ref, pred, classes=(0, 2, 8), iou_thr=0.5):
    stats = {}
    for c in classes:
        a = ref[ref[:, 5] == c] if len(ref) else ref
        b = pred[pred[:, 5] == c] if len(pred) else pred
        matched, deltas = 0, []
        used = np.zeros(len(b), dtype=bool)
        for box in a:
            if not len(b):
                break
            ious = iou_one(box[:4], b[:, :4])
            ious[used] = -1
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                matched += 1
                used[j] = True
                deltas.append(float(b[j, 4] - box[4]))
        stats[c] = {"ref": int(len(a)), "pred": int(len(b)), "matched": matched,
                    "match_rate": matched / max(len(a), 1),
                    "score_delta": float(np.mean(deltas)) if deltas else None}
    return stats


def prep(input_chw, bgr=False):
    arr = np.transpose(input_chw, (1, 2, 0))
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if bgr:
        arr = arr[:, :, ::-1].copy()
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--rknn", required=True)
    ap.add_argument("--precision", choices=("fp16", "int8"), required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--algorithm", default="normal")
    ap.add_argument("--method", default="channel")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    ref = np.load(args.ref)
    sids = [str(x) for x in ref["sids"]]
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    results = {"model": args.rknn, "precision": args.precision, "images": len(sids),
               "conf": args.conf}
    lc = {"inter": 0, "union": 0, "cw_i": 0, "cw_u": 0}
    st = {"ok": 0, "n": 0}
    dr = {"inter": np.zeros(3), "union": np.zeros(3)}
    det_stats = {0: [], 2: [], 8: []}
    all_counts = {"ref": 0, "pred": 0}

    try:
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                    target_platform="rk3566", quantized_algorithm=args.algorithm,
                    quantized_method=args.method)
        ret = rknn.load_onnx(model=args.onnx)
        if ret != 0:
            raise RuntimeError(f"load_onnx {ret}")
        if args.precision == "int8":
            ret = rknn.build(do_quantization=True, dataset=args.dataset)
        else:
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"build {ret}")
        ret = rknn.export_rknn(args.rknn)
        if ret != 0:
            raise RuntimeError(f"export {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime {ret}")

        for mode in ("rgb", "bgr"):
            lc = {"inter": 0, "union": 0, "cw_i": 0, "cw_u": 0}
            st = {"ok": 0, "n": 0}
            dr = {"inter": np.zeros(3), "union": np.zeros(3)}
            det_stats = {0: [], 2: [], 8: []}
            all_counts = {"ref": 0, "pred": 0}
            for i, sid in enumerate(sids):
                arr = prep(ref["images"][i], bgr=(mode == "bgr"))
                outs = [np.asarray(o) for o in rknn.inference(inputs=[arr], data_format="nhwc")]
                if i == 0:
                    results[f"{mode}_outputs"] = [
                        {"shape": list(o.shape), "dtype": str(o.dtype)} for o in outs]
                det = decode_raw(outs[:12])[0]
                pred_d = dets_from_decoded(det, args.conf)
                ref_d = dets_from_decoded(ref["det"][i], args.conf)
                all_counts["ref"] += len(ref_d)
                all_counts["pred"] += len(pred_d)
                for c, s in match_dets(ref_d, pred_d).items():
                    det_stats[c].append(s)

                p_lane = outs[12][0].argmax(0)
                p_style = outs[13][0].argmax(0)
                p_drv = outs[14][0].argmax(0)
                r_lane = ref["lane_cat"][i].argmax(0)
                r_style = ref["lane_style"][i].argmax(0)
                r_drv = ref["drivable"][i].argmax(0)
                pf, rf = p_lane != 8, r_lane != 8
                lc["inter"] += int((pf & rf).sum())
                lc["union"] += int((pf | rf).sum())
                pc, rc = p_lane == 0, r_lane == 0
                lc["cw_i"] += int((pc & rc).sum())
                lc["cw_u"] += int((pc | rc).sum())
                valid = r_style != 2
                st["ok"] += int(((p_style == r_style) & valid).sum())
                st["n"] += int(valid.sum())
                for c in range(3):
                    a, b = p_drv == c, r_drv == c
                    dr["inter"][c] += int((a & b).sum())
                    dr["union"][c] += int((a | b).sum())

            results[f"{mode}_seg"] = {
                "lane_fg_iou": lc["inter"] / max(lc["union"], 1),
                "crosswalk_iou": lc["cw_i"] / max(lc["cw_u"], 1),
                "style_acc_fg": st["ok"] / max(st["n"], 1),
                "drivable_miou": float(np.mean(dr["inter"] / np.maximum(dr["union"], 1))),
            }
            results[f"{mode}_det"] = {
                "counts": all_counts,
                "per_class": {str(k): {
                    "ref": int(sum(x["ref"] for x in v)),
                    "pred": int(sum(x["pred"] for x in v)),
                    "matched": int(sum(x["matched"] for x in v)),
                    "match_rate": sum(x["matched"] for x in v) / max(sum(x["ref"] for x in v), 1),
                    "score_delta": float(np.mean([x["score_delta"] for x in v if x["score_delta"] is not None]))
                    if any(x["score_delta"] is not None for x in v) else None,
                } for k, v in det_stats.items()},
            }
    finally:
        rknn.release()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
