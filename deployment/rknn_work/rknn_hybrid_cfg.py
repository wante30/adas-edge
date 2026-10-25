"""List or edit RKNN hybrid quantization cfg layers."""
from __future__ import annotations

import argparse
from pathlib import Path


def layers(cfg: Path) -> list[str]:
    names = []
    inside = False
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if line.startswith("quantize_parameters:"):
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            break
        if inside and line.startswith("    ") and not line.startswith("        ") and line.strip().endswith(":"):
            names.append(line.strip()[:-1])
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--float-tail", type=int, default=0)
    ap.add_argument("--float-prefix", action="append", default=[])
    ap.add_argument("--float-name", action="append", default=[])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    names = layers(args.cfg)
    if args.list:
        print("layers", len(names))
        for name in names[-60:]:
            print(name)
    if args.output:
        chosen = []
        if args.float_tail:
            chosen += names[-args.float_tail:]
        for name in names:
            if name in args.float_name:
                if name not in chosen:
                    chosen.append(name)
                continue
            if ("Constant" in name or name.endswith("_int8") or name == "images"
                    or "_i1" in name or name.endswith("-rs") or name.endswith("_sw")):
                continue
            if any(name.startswith(prefix) for prefix in args.float_prefix) or name in args.float_name:
                if name not in chosen:
                    chosen.append(name)
        text = args.cfg.read_text(encoding="utf-8")
        block = "custom_quantize_layers:\n" + "".join(f"    {name}: float16\n" for name in chosen)
        text = text.replace("custom_quantize_layers: {}", block.rstrip("\n"))
        args.output.write_text(text, encoding="utf-8")
        print("float layers", len(chosen), "->", args.output)


if __name__ == "__main__":
    main()
