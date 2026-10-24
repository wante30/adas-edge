"""Summarize B-05K/B-05L video timelines at multiple confidence thresholds."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(frames: list[dict], fps: float, conf: float) -> dict:
    flags = []
    colors = Counter()
    for frame in frames:
        items = [item for item in frame.get("detections", []) if float(item["confidence"]) >= conf]
        flags.append(bool(items))
        for item in items:
            colors[str(item.get("color", "UNKNOWN"))] += 1
    first = next((i for i, value in enumerate(flags) if value), None)
    runs = []
    start = None
    for index, value in enumerate(flags + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append({"start_s": round(start / fps, 3), "end_s": round((index - 1) / fps, 3),
                         "frames": index - start})
            start = None
    longest = max((run["frames"] for run in runs), default=0)
    return {
        "conf": conf,
        "frames": len(frames),
        "detection_frames": sum(flags),
        "frames_percent": sum(flags) / len(flags) if flags else 0.0,
        "first_frame": first,
        "first_seconds": None if first is None else round(first / fps, 3),
        "runs": len(runs),
        "longest_run_frames": longest,
        "colors": {color: colors.get(color, 0) for color in ("RED", "GREEN", "YELLOW", "UNKNOWN")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timelines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confs", type=float, nargs="+", default=[0.05, 0.25, 0.5])
    args = parser.parse_args()
    result = {}
    for path in sorted(args.timelines.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data["frames"]
        fps = float(data["summary"]["source_fps"])
        result[path.stem] = {str(conf): summarize(frames, fps, conf) for conf in args.confs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({video: {conf: {"frames": value[conf]["detection_frames"],
                                      "first_s": value[conf]["first_seconds"]}
                              for conf in value} for video, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
