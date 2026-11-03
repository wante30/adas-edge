"""Summarize the actual RK3566 render log; no simulated frames or timings."""
import json
import math
import sys
from pathlib import Path
rows = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
motion = [r for r in rows if r.get("road_mode") == "SCHEMATIC_ANIMATION"]
pairs = [(a,b) for a,b in zip(motion,motion[1:]) if a["road_moving"] and b["road_moving"]]
rates = [((b["road_offset"]-a["road_offset"]) % 36) / ((b["timestamp_ns"]-a["timestamp_ns"])/1e9) for a,b in pairs if 0 < b["timestamp_ns"]-a["timestamp_ns"] < 2000000000]
result = {"source":"RK3566 live render log", "samples":len(motion),
    "moving_samples":sum(r["road_moving"] for r in motion),
    "road_animation_relative_units_per_second":sum(rates)/len(rates) if rates else None,
    "scene_fps_mean":sum(r["scene_fps"] for r in motion)/len(motion) if motion else None,
    "max_live_tracks":max((r["live_tracks"] for r in motion),default=0),
    "note":"Animation rate is not measured vehicle speed. Recording can reduce UI FPS."}
assert len(motion)>=5 and len(rates)>=3, "Insufficient live evidence"
assert all(math.isfinite(r) and 0 < r < 7 for r in rates), "Road did not advance continuously"
print(json.dumps(result,ensure_ascii=False,indent=2))
