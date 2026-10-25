"""Convert an ONNX model to RKNN inside WSL (rknn-toolkit2 2.3.2)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def to_linux(path: str) -> Path:
    if path.startswith("/mnt/"):
        return Path(path)
    drive, rest = os.path.splitdrive(path)
    if drive:
        return Path("/mnt") / drive[0].lower() / rest.lstrip("\\/").replace("\\", "/")
    return Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--calib-dir")
    ap.add_argument("--precision", choices=("fp16", "int8"), default="fp16")
    ap.add_argument("--quantized-algorithm", default="normal")
    ap.add_argument("--probe-image")
    args = ap.parse_args()

    from rknn.api import RKNN

    onnx_path = to_linux(args.onnx)
    output = to_linux(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = None
    if args.precision == "int8":
        if not args.calib_dir:
            raise SystemExit("--calib-dir is required for int8")
        calib_dir = to_linux(args.calib_dir)
        images = sorted(calib_dir.glob("*.jpg"))
        if not images:
            raise SystemExit(f"no calibration images in {calib_dir}")
        dataset_file = output.parent / f"{output.stem}_calib.txt"
        dataset_file.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
        dataset = str(dataset_file)
        print("calib images", len(images), flush=True)

    rknn = RKNN(verbose=False)
    try:
        status = rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                             target_platform="rk3566", quantized_algorithm=args.quantized_algorithm,
                             quantized_method="channel")
        print("config", status, flush=True)
        print("load_onnx", rknn.load_onnx(model=str(onnx_path)), flush=True)
        print("build", rknn.build(do_quantization=args.precision == "int8", dataset=dataset), flush=True)
        print("export", rknn.export_rknn(str(output)), flush=True)
        if args.probe_image:
            import cv2
            image = to_linux(args.probe_image)
            bgr = cv2.imread(str(image))
            print("probe image", bgr.shape if bgr is not None else None, flush=True)
            rknn.init_runtime()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            outputs = rknn.inference(inputs=[rgb])
            print("probe outputs", [tuple(o.shape) for o in outputs], flush=True)
    finally:
        rknn.release()
    print("RKNN", args.precision, output, flush=True)


if __name__ == "__main__":
    main()
