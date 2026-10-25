"""Print a compact summary of a B-05O metrics.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path, nargs="+")
    args = ap.parse_args()
    for path in args.metrics:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"== {path.parent.name}/{data['mode']} {Path(data['video']).name}")
        print(f"frames={data['frames']} fps={data['fps']:.2f} duration={data['duration_s']:.2f}s")
        for key in ("bdd", "p2", "spec"):
            r = data["rates"][key]
            print(f"  {key:4s} count={r['count']:5d} hz={r['hz']:.2f} mean={r['mean_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")
        print("  scheme fwd/s:", {k: round(v, 2) for k, v in data["scheme_total_forward_per_second"].items()})
        for key, value in data.get("display_stats", {}).items():
            print(f"  display {key}: {value}")
        for key, value in data.get("tl_tracks", {}).items():
            print(f"  tracks {key}: {value}")
        for event in data.get("key_events", []):
            boxes = [(d["color"], round(d["confidence"], 3), [round(v, 1) for v in d["bbox"]]) for d in event["detections"]]
            print(f"  key t={event['t']} {event['scheme']}: {event['tl_count']} {boxes}")


if __name__ == "__main__":
    main()
