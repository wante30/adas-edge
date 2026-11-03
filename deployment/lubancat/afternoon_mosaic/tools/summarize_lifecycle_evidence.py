"""Summarize renderer continuity, not object recognition or ground-truth ID accuracy."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stats(values):
    if not values:
        return {}
    values = sorted(values)
    def percentile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"median": statistics.median(values), "p99": percentile(.99), "max": values[-1]}


def summarize(trace, scene):
    rows = read(trace)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source_key"], row["track_id"])].append(row)
    visible = {key: [r for r in values if r["fade"] > .05] for key, values in groups.items()}
    visible = {key: values for key, values in visible.items() if values}
    speeds, steps60, recovery_jumps = [], [], []
    for key, values in groups.items():
        for a, b in zip(values, values[1:]):
            dt = (b["render_ns"] - a["render_ns"]) / 1e9
            if not 0 < dt < .2 or b["birth_count"] != a["birth_count"]:
                continue
            if a["fade"] > .05 and b["fade"] > .05:
                speed = math.hypot(b["filtered_x"] - a["filtered_x"],
                                   b["filtered_z"] - a["filtered_z"]) / dt
                speeds.append(speed)
                steps60.append(speed / 60)
            # Fade-in is allowed up to 5 alpha units/s; count faster recoveries.
            if (a["measurement_age_s"] > .55 and b["measurement_age_s"] < a["measurement_age_s"]
                    and b["fade"] - a["fade"] > 5.1 * dt):
                recovery_jumps.append({"track_id": key[1], "dt": dt,
                                       "from": a["fade"], "to": b["fade"]})
    scene_rows = read(scene)
    return {
        "trace_rows": len(rows),
        "visible_display_id_count": len(visible),
        "visible_classes": dict(Counter(v[0].get("class", "not_logged") for v in visible.values())),
        "visible_track_duration_s": stats([(v[-1]["render_ns"] - v[0]["render_ns"]) / 1e9 for v in visible.values()]),
        "server_id_reassociations": sum(max(r.get("reassociated_count", 0) for r in v) for v in groups.values()),
        "nonpositive_server_id_rows": sum(r["server_track_id"] <= 0 for r in rows),
        "opacity_recovery_jumps": len(recovery_jumps),
        "opacity_recovery_examples": recovery_jumps[:5],
        "speed_relative_units_per_s": stats(speeds),
        "step_normalized_to_60hz": stats(steps60),
        "model_height_scales": sorted(set(r["model_height_scale"] for r in rows if "model_height_scale" in r)),
        "scene_fps": stats([r["scene_fps"] for r in scene_rows]),
        "coasting_rows_after_600ms": sum(r["measurement_age_s"] > .6 for r in rows),
        "note": "Renderer-only replay; fewer displayed IDs is not detection accuracy. Coordinates are relative units. FPS includes concurrent X11 recording and is not whole-HMI throughput."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {label: summarize(root / (label + "_trace.jsonl"), root / (label + "_scene.jsonl"))
              for label, root in [("before", args.before), ("after", args.after)]}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
