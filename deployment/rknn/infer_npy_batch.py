"""Run an RKNN model over pre-letterboxed NHWC canvases and save the raw outputs.

Runs on the RK3566. Inputs come from ``tools/rect_canvas_dump.py`` on the host, so
this script never letterboxes anything itself and the board sees byte-identical
inputs to the PC backends.

Writes one ``.npz`` per canvas plus a ``board_infer.json`` with the per-image
inference time, the output tensor shapes and the process RSS - enough for both the
INT8 correctness comparison and the single-model benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        help="directory holding the .npy canvases; overrides the host paths recorded in the manifest",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 means every entry in the manifest")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    if args.limit > 0:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit("manifest contains no entries")

    def canvas_path(entry: dict) -> Path:
        """Resolve a manifest entry, preferring --inputs-dir over the host path."""
        if args.inputs_dir is not None:
            candidate = Path(args.inputs_dir) / entry["file"]
            if not candidate.is_file():
                raise FileNotFoundError(f"{candidate} is missing; run tools/rect_canvas_dump.py on the host first")
            return candidate
        return Path(entry["path"])

    try:
        import psutil

        process = psutil.Process(os.getpid())
    except ImportError:
        process = None

    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        if rknn.load_rknn(str(args.model)) != 0:
            raise RuntimeError(f"load_rknn failed for {args.model}")
        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError("init_runtime failed")
        rss_after_init = process.memory_info().rss if process else None
        for index in range(min(args.warmup, len(entries))):
            canvas = np.load(canvas_path(entries[index]))
            rknn.inference(inputs=[canvas], data_format="nhwc")
        for entry in entries:
            canvas = np.load(canvas_path(entry))
            if canvas.dtype != np.uint8 or canvas.ndim != 4 or canvas.shape[-1] != 3:
                raise RuntimeError(f"{canvas_path(entry)}: expected NHWC uint8, got {canvas.shape} {canvas.dtype}")
            started = time.perf_counter()
            outputs = rknn.inference(inputs=[canvas], data_format="nhwc")
            elapsed_ms = (time.perf_counter() - started) * 1000
            destination = args.out_dir / f"{entry['stem']}_out.npz"
            np.savez_compressed(destination, **{f"out{i}": np.asarray(output) for i, output in enumerate(outputs)})
            rows.append(
                {
                    "stem": entry["stem"],
                    "scale": entry["scale"],
                    "pad_x": entry["pad_x"],
                    "pad_y": entry["pad_y"],
                    "inference_ms": elapsed_ms,
                    "outputs": [{"shape": list(np.asarray(o).shape), "dtype": str(np.asarray(o).dtype)} for o in outputs],
                }
            )
        rss_after_runs = process.memory_info().rss if process else None
    finally:
        rknn.release()

    values = np.asarray([row["inference_ms"] for row in rows])
    report = {
        "model": str(args.model),
        "manifest": str(args.manifest),
        "canvas": manifest["canvas"],
        "layout": manifest["layout"],
        "channel_order": manifest["channel_order"],
        "input_array_shape": manifest["array_shape"],
        "entries": len(rows),
        "warmup": args.warmup,
        "npu_only_ms": {
            "mean": round(float(values.mean()), 3),
            "p50": round(float(np.percentile(values, 50)), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
            "min": round(float(values.min()), 3),
            "max": round(float(values.max()), 3),
        },
        "npu_only_fps": round(1000.0 / float(values.mean()), 3),
        "process_rss_bytes": {"after_init": rss_after_init, "after_runs": rss_after_runs},
        "rows": rows,
    }
    destination = args.out_dir / "board_infer.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
