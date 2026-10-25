"""Inspect RKNN quantization cfg (no PyYAML dependency)."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse(path: Path) -> dict[str, dict]:
    params: dict[str, dict] = {}
    current = None
    pending = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("quantize_parameters:"):
            continue
        if raw.startswith("    ") and not raw.startswith("        ") and raw.rstrip().endswith(":"):
            current = raw.strip()[:-1]
            params[current] = {}
            pending = None
            continue
        if current is None:
            continue
        if raw.startswith("        ") and not raw.startswith("            "):
            key, _, value = raw.strip().partition(":")
            value = value.strip()
            if value:
                params[current][key] = value
                pending = None
            else:
                params[current][key] = []
                pending = key
            continue
        if raw.lstrip().startswith("- ") and pending:
            params[current][pending].append(raw.lstrip()[2:].strip())
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    args = ap.parse_args()
    params = parse(args.cfg)
    print("layers", len(params))
    print("dtypes", Counter(v.get("dtype") for v in params.values()))
    print("qmethods", Counter(v.get("qmethod") for v in params.values()))
    for name in ("images", "boxes", "scores", "boxes_int8", "scores_int8"):
        if name in params:
            print(name, params[name])
    for name in list(params)[-10:]:
        print("tail", name, {k: params[name].get(k) for k in ("dtype", "min", "max", "scale", "zero_point")})


if __name__ == "__main__":
    main()
