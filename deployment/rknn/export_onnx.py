"""Export a fixed-shape Ultralytics YOLO checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=str, default="640", help="square integer or WxH, e.g. 512x288")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    size_text = args.imgsz.lower().replace("×", "x")
    if "x" in size_text:
        width, height = (int(part) for part in size_text.split("x", 1))
        export_size: int | tuple[int, int] = (height, width)
    else:
        export_size = int(size_text)

    exported = Path(YOLO(str(args.weights)).export(
        format="onnx", imgsz=export_size, opset=args.opset, dynamic=False, simplify=True
    ))
    if exported.resolve() != args.output.resolve():
        args.output.write_bytes(exported.read_bytes())
    print(f"ONNX exported: {args.output}")


if __name__ == "__main__":
    main()
