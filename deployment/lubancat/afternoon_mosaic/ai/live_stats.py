import json
import sys
from collections import deque

path = "/home/cat/qt_gpu_probe_ai_20261019/results/adas_B_live_default_timing.jsonl"
count = int(sys.argv[1]) if len(sys.argv) > 1 else 600
rows = []
for line in deque(open(path, errors="replace"), maxlen=count):
    try:
        row = json.loads(line)
    except Exception:
        continue
    if row.get("result_publish_timestamp_ns"):
        rows.append(row)
if len(rows) < 2:
    raise SystemExit("not enough rows")
rows.sort(key=lambda r: r["result_publish_timestamp_ns"])
first, last = rows[0], rows[-1]
span = (last["result_publish_timestamp_ns"] - first["result_publish_timestamp_ns"]) / 1e9
main = [r for r in rows if r.get("main_ran")]


def stats(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    return (round(values[len(values) // 2], 2),
            round(values[min(len(values) - 1, int(len(values) * 0.95))], 2),
            round(sum(values) / len(values), 2))


print("rows", len(rows), "span_s", round(span, 2),
      "result_fps", round(len(rows) / span, 2) if span else None,
      "main_fps", round(len(main) / span, 2) if span else None)
for key in ("total_ms", "rknn_call_ms", "decode_nms_ms", "preprocess_ms"):
    print(" main", key, stats([r["main_timing"].get(key) for r in main if r.get("main_timing")]))
print(" main_ms", stats([r.get("main_ms") for r in main]))
print(" pipeline_ms", stats([r.get("pipeline_ms") for r in main]))
print(" decode_ms", stats([r.get("decode_ms") for r in main]))
print(" boxes", stats([r.get("boxes") for r in main]),
      "counts", stats([r.get("main_count") for r in main]))
print(" last_frame", last.get("frame_id"), "last_boxes", last.get("boxes"),
      "last_counts", last.get("detection_counts"), "traffic", last.get("traffic"))
