"""Build an RKNN model from ONNX and validate it in the Toolkit2 simulator."""

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
        sx, sy = w / ww, h / hh
        gx = np.tile(np.arange(ww, dtype=np.float32), hh)
        gy = np.repeat(np.arange(hh, dtype=np.float32), ww)
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


def prep(input_chw, bgr=False):
    arr = np.transpose(input_chw, (1, 2, 0))
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if bgr:
        arr = arr[:, :, ::-1].copy()
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--rknn", required=True, help="output .rknn path")
    ap.add_argument("--precision", choices=("fp16", "int8"), required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--algorithm", default="normal")
    ap.add_argument("--method", default="channel")
    args = ap.parse_args()

    ref = np.load(args.ref)
    sids = [str(x) for x in ref["sids"]]
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    results = {"model": args.rknn, "precision": args.precision, "images": len(sids)}
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
            seg_agree, det_max = [], []
            for i, sid in enumerate(sids):
                arr = prep(ref["images"][i], bgr=(mode == "bgr"))
                outs = [np.asarray(o) for o in rknn.inference(inputs=[arr], data_format="nhwc")]
                if i == 0:
                    results[f"{mode}_outputs"] = [
                        {"shape": list(o.shape), "dtype": str(o.dtype)} for o in outs]
                det = decode_raw(outs[:12])[0]
                ref_seg = [ref[k][i] for k in ("lane_cat", "lane_style", "drivable")]
                seg_pred = [o[0].argmax(0) for o in outs[12:]]
                ref_pred = [a.argmax(0) for a in ref_seg]
                seg_agree.append([float((p == r).mean()) for p, r in zip(seg_pred, ref_pred)])
                det_max.append(float(np.abs(det[:4] - ref["det"][i][:4]).max()))
            results[f"{mode}_seg_pixel_agreement"] = np.mean(np.asarray(seg_agree), 0).tolist()
            results[f"{mode}_seg_agreement_min"] = np.min(np.asarray(seg_agree), 0).tolist()
            results[f"{mode}_det_xywh_max_diff"] = float(max(det_max))
    finally:
        rknn.release()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
