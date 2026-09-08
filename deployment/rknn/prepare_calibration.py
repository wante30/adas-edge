"""Build a deterministic representative-image manifest for RKNN INT8 calibration."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--linux-root", type=str, help="replace input-dir with this path in output manifest")
    args = parser.parse_args()
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(path for path in args.input_dir.rglob("*") if path.suffix.lower() in suffixes)[:args.limit]
    if not images:
        raise FileNotFoundError(f"no images found below {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.linux_root:
        root = args.linux_root.rstrip("/")
        lines = [f"{root}/{path.relative_to(args.input_dir).as_posix()}" for path in images]
    else:
        lines = [str(path.resolve()) for path in images]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(images)} calibration paths to {args.output}")


if __name__ == "__main__":
    main()
