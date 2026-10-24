"""Strip per-image predictions from a B-05L evaluation JSON for Git delivery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    data.pop("predictions_by_image", None)
    data.pop("input_sha256_by_image", None)
    if "color" in data:
        data["color"].pop("by_image", None)
    pr = data.get("curves", {}).get("pr50", {})
    if "confidence" in pr:
        pr["confidence"] = pr["confidence"][:200]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"input": str(args.input.resolve()), "output": str(args.output.resolve()),
                      "size_bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
