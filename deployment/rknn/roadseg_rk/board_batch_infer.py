"""Batch-run an RKNN model on the RK3566 and save raw outputs for PC comparison.

The manifest is a JSON list of {"name": "...", "input": "/path/to.npy"}.
Each input is NHWC uint8 RGB [1,192,320,3].
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(args.model)) != 0:
        raise RuntimeError("load_rknn failed")
    if rknn.init_runtime() != 0:
        raise RuntimeError("init_runtime failed")

    first = np.load(rows[0]["input"])
    if first.ndim == 3:
        first = first[None, ...]
    for _ in range(args.warmup):
        rknn.inference(inputs=[first], data_format="nhwc")

    results = []
    for row in rows:
        image = np.load(row["input"])
        if image.ndim == 3:
            image = image[None, ...]
        if image.dtype != np.uint8 or image.shape[-1] != 3:
            raise ValueError("bad input %s: %r %s" % (row["input"], image.shape, image.dtype))
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[image], data_format="nhwc")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        arr = np.asarray(outputs[0], dtype=np.float32)
        out_path = args.out_dir / (row["name"] + ".npy")
        np.save(str(out_path), arr)
        results.append(
            {
                "name": row["name"],
                "input": row["input"],
                "output": str(out_path),
                "latency_ms": round(latency_ms, 3),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "finite": bool(np.isfinite(arr).all()),
                "channel_mean": [float(arr[0, c].mean()) for c in range(arr.shape[1])]
                if arr.ndim == 4
                else None,
            }
        )
    rknn.release()

    payload = {"model": str(args.model), "count": len(results), "rows": results}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
