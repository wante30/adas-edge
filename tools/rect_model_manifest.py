"""Collect the build evidence for the rectangular ONNX / RKNN model set.

Produces one JSON (and a Markdown table) recording, per canvas:

* SHA-256 of the source checkpoint, the static ONNX, the split-output ONNX and the
  RKNN model;
* the ONNX input and output tensor shapes, so the frozen canvas is provable from
  the artifact itself rather than from the command that produced it;
* the calibration provenance, including the sample count, the source material and
  the fact that the frozen holdout was not used;
* the quantization configuration the RKNN build used.

Nothing here runs a model; it only reads artifacts and hashes them, so it is safe
to re-run as an audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python tools/rect_model_manifest.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHUNK = 1 << 20

# Must match deployment/rknn/convert_rknn.py; recorded here so an audit can see
# exactly what the INT8 scales were computed under.
QUANT_CONFIG = {
    "mean_values": [[0, 0, 0]],
    "std_values": [[255, 255, 255]],
    "target_platform": "rk3566",
    "do_quantization": True,
    "toolkit": "rknn-toolkit2 2.3.2",
    "python": "3.10",
    "onnx": "1.16.1",
    "protobuf": "3.20.3",
}


def sha256_file(path: Path) -> str | None:
    """SHA-256 of a file, or None when it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest().upper()


def onnx_io_shapes(path: Path) -> dict | None:
    """Input and output tensor shapes of a static ONNX graph."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        import onnx
    except ImportError:  # pragma: no cover - onnx is a dev dependency
        return None
    model = onnx.load(str(path))
    def shapes(values):
        result = []
        for value in values:
            dims = []
            for dimension in value.type.tensor_type.shape.dim:
                dims.append(dimension.dim_value if dimension.HasField("dim_value") else str(dimension.dim_param))
            result.append({"name": value.name, "shape": dims, "dynamic": any(isinstance(d, str) for d in dims)})
        return result
    return {"inputs": shapes(model.graph.input), "outputs": shapes(model.graph.output)}


def collect(models_dir: Path, weights: Path, canvases: list[tuple[int, int]]) -> dict:
    """Assemble the evidence record for every canvas."""
    entries = []
    for width, height in canvases:
        canvas = f"{width}x{height}"
        onnx_path = models_dir / f"yolo11n_{canvas}.onnx"
        split_path = models_dir / f"yolo11n_{canvas}_split.onnx"
        rknn_path = models_dir / f"yolo11n_rk3566_int8_split_{canvas}.rknn"
        calibration_path = models_dir / f"calib_{canvas}.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.is_file() else None
        entries.append(
            {
                "canvas": canvas,
                "canvas_pixels": width * height,
                "static_onnx": {
                    "path": str(onnx_path),
                    "sha256": sha256_file(onnx_path),
                    "io": onnx_io_shapes(onnx_path),
                },
                "split_onnx": {
                    "path": str(split_path),
                    "sha256": sha256_file(split_path),
                    "io": onnx_io_shapes(split_path),
                },
                "rknn_int8": {
                    "path": str(rknn_path),
                    "sha256": sha256_file(rknn_path),
                    "present": rknn_path.is_file(),
                },
                "calibration": (
                    {
                        "manifest": calibration.get("manifest"),
                        "samples": calibration.get("samples"),
                        "sources": calibration.get("sources"),
                        "channel_order": calibration.get("channel_order"),
                        "layout": calibration.get("layout"),
                        "array_shape": calibration.get("array_shape"),
                        "frozen_holdout_used": calibration.get("frozen_holdout_used"),
                    }
                    if calibration
                    else None
                ),
            }
        )
    return {
        "weights": {"path": str(weights), "sha256": sha256_file(weights)},
        "quant_config": QUANT_CONFIG,
        "canvases": entries,
    }


def render_markdown(record: dict) -> str:
    lines = [
        "# Rectangular model build manifest",
        "",
        f"- source checkpoint: `{record['weights']['path']}`",
        f"- checkpoint SHA-256: `{record['weights']['sha256']}`",
        f"- toolkit: `{record['quant_config']['toolkit']}` (python {record['quant_config']['python']}, "
        f"onnx {record['quant_config']['onnx']}, protobuf {record['quant_config']['protobuf']})",
        f"- mean/std: `{record['quant_config']['mean_values']}` / `{record['quant_config']['std_values']}`, "
        f"target `{record['quant_config']['target_platform']}`",
        "",
        "| canvas | px | static ONNX in/out | split ONNX out | ONNX sha256 | RKNN INT8 sha256 | calib samples |",
        "| --- | ---: | --- | --- | --- | --- | ---: |",
    ]
    for entry in record["canvases"]:
        static_io = entry["static_onnx"]["io"]
        split_io = entry["split_onnx"]["io"]
        static_desc = "n/a"
        if static_io:
            in_shape = static_io["inputs"][0]["shape"]
            out_shapes = " + ".join(str(item["shape"]) for item in static_io["outputs"])
            static_desc = f"{in_shape} -> {out_shapes}"
        split_desc = "n/a"
        if split_io:
            split_desc = " + ".join(f"{item['name']}{item['shape']}" for item in split_io["outputs"])
        calibration = entry["calibration"] or {}
        lines.append(
            f"| {entry['canvas']} | {entry['canvas_pixels']} | {static_desc} | {split_desc} | "
            f"`{(entry['static_onnx']['sha256'] or 'missing')[:16]}...` | "
            f"`{(entry['rknn_int8']['sha256'] or 'missing')[:16]}...` | {calibration.get('samples', 'n/a')} |"
        )
    missing = [entry["canvas"] for entry in record["canvases"] if not entry["rknn_int8"]["present"]]
    lines += ["", f"- RKNN models missing: {missing if missing else 'none'}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--canvases", default="320x192,416x256,512x288,640x352")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    canvases = []
    for text in args.canvases.split(","):
        text = text.strip()
        if not text:
            continue
        width, height = (int(part) for part in text.lower().split("x", 1))
        canvases.append((width, height))

    record = collect(args.models_dir, args.weights, canvases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    table = render_markdown(record)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
