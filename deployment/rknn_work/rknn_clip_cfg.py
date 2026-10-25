"""Rewrite output-layer min/max in an RKNN hybrid cfg to clip outliers."""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_block(lines: list[str], name: str, block: list[str]) -> list[str]:
    start = None
    for index, line in enumerate(lines):
        if line.strip() == f"{name}:" and line.startswith("    ") and not line.startswith("        "):
            start = index
            break
    if start is None:
        raise SystemExit(f"layer {name} not found")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("    ") and not lines[index].startswith("        ") and lines[index].rstrip().endswith(":"):
            end = index
            break
    return lines[:start] + block + lines[end:]


def block_for(name: str, dtype: str, minimum: float, maximum: float) -> list[str]:
    scale = (maximum - minimum) / 255.0
    zero_point = round(-128 - minimum / scale)
    return [
        f"    {name}:",
        "        qtype: asym",
        "        qmethod: layer",
        f"        dtype: {dtype}",
        "        min:",
        f"        -   {minimum}",
        "        max:",
        f"        -   {maximum}",
        "        scale:",
        f"        -   {scale}",
        "        zero_point:",
        f"        -   {zero_point}",
        "        ori_min:",
        f"        -   {minimum}",
        "        ori_max:",
        f"        -   {maximum}",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--box-min", type=float, default=-32.0)
    ap.add_argument("--box-max", type=float, default=544.0)
    ap.add_argument("--score-min", type=float, default=-12.0)
    ap.add_argument("--score-max", type=float, default=6.0)
    args = ap.parse_args()
    lines = args.cfg.read_text(encoding="utf-8").splitlines()
    lines = replace_block(lines, "boxes_int8", block_for("boxes_int8", "int8", args.box_min, args.box_max))
    lines = replace_block(lines, "scores_int8", block_for("scores_int8", "int8", args.score_min, args.score_max))
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("written", args.output)


if __name__ == "__main__":
    main()
