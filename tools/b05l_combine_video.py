"""Combine per-model B-05L video summaries into one timeline JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timelines", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for label in args.labels:
        for path in sorted((args.timelines / label).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            summary = data["summary"]
            summary["label"] = label
            summaries.append(summary)
    payload = {
        "schema": "B05L-video-detection-timeline-v1",
        "source": "four official competition videos, 1920x1080, every-frame inference, conf=0.05, NMS IoU=0.5",
        "labels": args.labels,
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "summaries": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
