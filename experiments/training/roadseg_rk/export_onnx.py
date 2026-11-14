"""Export a RoadSeg-RK checkpoint to static ONNX and verify ONNX parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .eval_roadseg import load_model
from .roadseg_data import RoadSegDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--parity-samples", type=int, default=8)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model, ck, layout, size, names = load_model(args.checkpoint, "cpu")
    example = torch.zeros(1, 3, size[1], size[0])
    onnx_path = out / "model.fp32.onnx"
    torch.onnx.export(model, example, str(onnx_path), input_names=["rgb"], output_names=["logits"],
                      opset_version=args.opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ds = RoadSegDataset(args.index, "val", augment=False, size=size, seed=15)
    ids = sorted(range(len(ds)), key=lambda i: hashlib.sha256(ds.rows[i]["id"].encode()).hexdigest())[:args.parity_samples]
    errors = []
    for i in ids:
        x, _, _ = ds[i]
        with torch.no_grad():
            ref = model(x[None]).numpy()
        got = session.run(None, {"rgb": x[None].numpy()})[0]
        errors.append(float(np.max(np.abs(ref - got))))
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  model=ck["config"]["model"], epoch=ck.get("epoch"), layout=layout,
                  input_shape=[1, 3, size[1], size[0]], classes=names,
                  onnx_bytes=onnx_path.stat().st_size,
                  onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
                  parity_max_abs=errors, parity_max=float(max(errors)) if errors else None,
                  opset=args.opset)
    (out / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
