"""Export the B-1 multi-task model to a static ONNX graph (raw detection heads + 3 seg heads)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mtl.model import build_model  # noqa: E402

NAMES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike",
         "traffic light", "traffic sign"]
LANE_NAMES = ["crosswalk", "double other", "double white", "double yellow", "road curb",
              "single other", "single white", "single yellow", "background"]
STYLE_NAMES = ["solid", "dashed", "background"]
DRIVABLE_NAMES = ["direct", "alternative", "background"]


def _as_nchw(value, channels, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(f"{name} must be rank 4, got {array.shape}")
    if array.shape[1] == channels:
        return array
    if array.shape[-1] == channels:
        return array.transpose(0, 3, 1, 2)
    raise ValueError(f"{name} has no channel dimension of size {channels}: {array.shape}")


def decode_raw_heads(outputs, input_height=288, input_width=512):
    if len(outputs) < 3 or len(outputs) % 3:
        raise ValueError(f"expected three outputs per scale, got {len(outputs)}")
    boxes_all, scores_all = [], []
    reg_max = None
    for scale in range(len(outputs) // 3):
        box = _as_nchw(outputs[scale * 3], 64, f"scale {scale} reg")
        if box.shape[1] % 4:
            raise ValueError(f"invalid reg channels {box.shape}")
        cur = box.shape[1] // 4
        reg_max = cur if reg_max is None else reg_max
        if cur != reg_max:
            raise ValueError("mixed reg_max")
        h, w = box.shape[2:]
        scores = np.asarray(outputs[scale * 3 + 1], dtype=np.float32)
        if scores.ndim == 3:
            scores = scores[None]
        if scores.shape[2:] != (h, w):
            raise ValueError(f"scores shape {scores.shape} != {(h, w)}")
        logits = box.reshape(box.shape[0], 4, reg_max, h * w)
        logits = logits - logits.max(axis=2, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=2, keepdims=True)
        bins = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)
        dist = (prob * bins).sum(axis=2)
        sx, sy = input_width / w, input_height / h
        gx = np.tile(np.arange(w, dtype=np.float32), h)
        gy = np.repeat(np.arange(h, dtype=np.float32), w)
        left = (gx[None] + 0.5 - dist[:, 0]) * sx
        top = (gy[None] + 0.5 - dist[:, 1]) * sy
        right = (gx[None] + 0.5 + dist[:, 2]) * sx
        bottom = (gy[None] + 0.5 + dist[:, 3]) * sy
        boxes_all.append(np.stack(((left + right) / 2, (top + bottom) / 2,
                                   right - left, bottom - top), axis=1))
        scores_all.append(np.clip(scores.reshape(scores.shape[0], scores.shape[1], -1), 0, 1))
    return np.concatenate((np.concatenate(boxes_all, 2), np.concatenate(scores_all, 2)), 1)


class MTLExport(nn.Module):
    """Deterministic export graph: neck taps -> raw detect branches + shared seg decoder."""

    def __init__(self, mtl):
        super().__init__()
        self.det = mtl.det
        self.decoder = mtl.decoder
        self.head = mtl.det.model[-1]
        self.head_f = [int(i) for i in self.head.f]
        self.save = set(int(i) for i in self.det.save)

    def features(self, x):
        y = []
        for m in self.det.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return [y[i] for i in self.head_f]

    def forward(self, x):
        feats = self.features(x)
        outs = []
        for i in range(self.head.nl):
            reg = self.head.cv2[i](feats[i])
            cls = self.head.cv3[i](feats[i]).sigmoid()
            total = cls.sum(1, keepdim=True).clamp(0, 1)
            outs.extend((reg, cls, total))
        seg = self.decoder(feats[0], feats[1], feats[2])
        return tuple(outs + [seg["lane_cat"], seg["lane_style"], seg["drivable"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/bdd100k")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--det-ckpt", default="weights/b05n_p2_multiclass.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--ids", default="c962fce0-dcea4e6a,c8a7a62d-8d489d6e,bd1ae0c2-325d1820,"
                                      "c78f9fc6-024da644,b7390768-06a6a4a0,c5568343-7dfa0efd,"
                                      "c33fd06b-7beab8d7,bd95af88-2228e86b,c75e23fe-fbadb3fa,"
                                      "b9d9ed4e-66448e19")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parity = out.parent / "parity"
    (parity / "images").mkdir(parents=True, exist_ok=True)

    model = build_model(args.det_ckpt, device="cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    wrapper = MTLExport(model).eval()

    ids = [s for s in args.ids.split(",") if s]
    inputs, dets, raw_list, segs = [], [], [], {"lane_cat": [], "lane_style": [], "drivable": []}
    img_dir = Path(args.root) / "images" / "flat" / "val"
    from PIL import Image
    import shutil

    for sid in ids:
        src = img_dir / f"{sid}.jpg"
        img = Image.open(src).convert("RGB").resize((512, 288), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1)).float()[None]
        with torch.no_grad():
            seg, det, _ = model(x, run_det=True)
            raw = wrapper(x)
            y = det[0] if isinstance(det, tuple) else det
        inputs.append(x.numpy()[0])
        dets.append(y.numpy()[0])
        raw_list.append([r.numpy() for r in raw[:12]])
        for k in segs:
            segs[k].append(seg[k].numpy()[0])
        shutil.copy(src, parity / "images" / f"{sid}.jpg")
        print("ref", sid, flush=True)

    decoded = decode_raw_heads(raw_list[0], 288, 512)
    ref_decoded = dets[0][None]
    print("raw decode diff", float(np.abs(decoded - ref_decoded).max()), flush=True)

    names = []
    for i in range(4):
        for kind in ("reg", "cls", "cls_sum"):
            names.append(f"{kind}_s{[4, 8, 16, 32][i]}")
    names += ["lane_cat", "lane_style", "drivable"]

    sample = torch.zeros(1, 3, 288, 512)
    with torch.no_grad():
        torch.onnx.export(wrapper, sample, str(out), opset_version=args.opset,
                          input_names=["images"], output_names=names,
                          dynamic_axes=None, do_constant_folding=True)
    print("onnx written", out, out.stat().st_size, flush=True)

    import onnx
    graph = onnx.load(str(out))
    onnx.checker.check_model(graph)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: inputs[0][None]})

    diffs = []
    for name, r, o in zip(names[:12], raw_list[0], ort_out[:12]):
        diffs.append({"name": name, "shape": list(o.shape),
                      "max_abs_diff": float(np.abs(r - o).max())})
    seg_diffs = []
    for name, r, o in zip(names[12:], [segs["lane_cat"][0], segs["lane_style"][0], segs["drivable"][0]],
                          ort_out[12:]):
        seg_diffs.append({"name": name, "shape": list(o.shape),
                          "max_abs_diff": float(np.abs(r - o).max())})

    ort_decoded = decode_raw_heads(ort_out[:12], 288, 512)
    det_diff = float(np.abs(ort_decoded - ref_decoded).max())
    print("ORT raw max diff", max(d["max_abs_diff"] for d in diffs), flush=True)
    print("ORT decoded diff", det_diff, flush=True)
    print("ORT seg diff", max(s["max_abs_diff"] for s in seg_diffs), flush=True)

    np.savez_compressed(parity / "reference.npz",
                        images=np.stack(inputs), det=np.stack(dets),
                        lane_cat=np.stack(segs["lane_cat"]).astype(np.float16),
                        lane_style=np.stack(segs["lane_style"]).astype(np.float16),
                        drivable=np.stack(segs["drivable"]).astype(np.float16),
                        sids=np.asarray(ids))

    contract = {
        "model": "b05n_p2_mtl_b1",
        "input": {"name": "images", "shape": [1, 3, 288, 512], "layout": "NCHW",
                  "color": "RGB", "dtype": "float32", "scale": "0..1 (raw/255)"},
        "detection_raw_heads": {
            "ordering": "per scale: reg, cls, cls_sum; scales s4,s8,s16,s32",
            "reg": {"shape": [1, 64, "H", "W"], "meaning": "DFL logits, 4 sides x 16 bins"},
            "cls": {"shape": [1, 10, "H", "W"], "meaning": "sigmoid class probabilities"},
            "cls_sum": {"shape": [1, 1, "H", "W"], "meaning": "sum of class probabilities"},
            "classes": NAMES,
            "postprocess": "host DFL softmax/expectation -> ltrb -> xywh -> NMS",
        },
        "segmentation": {
            "lane_cat": {"shape": [1, 9, 288, 512], "classes": LANE_NAMES, "activation": "logits"},
            "lane_style": {"shape": [1, 3, 288, 512], "classes": STYLE_NAMES, "activation": "logits"},
            "drivable": {"shape": [1, 3, 288, 512], "classes": DRIVABLE_NAMES, "activation": "logits"},
        },
        "output_names": names,
        "onnx_parity": {"raw": diffs, "seg": seg_diffs, "decoded_det_max_abs_diff": det_diff},
    }
    (out.parent / "output_contract.json").write_text(json.dumps(contract, indent=2))
    print("contract written", flush=True)


if __name__ == "__main__":
    main()
