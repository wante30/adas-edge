import json
import sys
from collections import deque

path = "/home/cat/qt_gpu_probe_ai_20261019/results/adas_B_live_default_timing.jsonl"
count = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rows = []
for line in deque(open(path, errors="replace"), maxlen=count):
    try:
        row = json.loads(line)
    except Exception:
        continue
    if row.get("result_publish_timestamp_ns") and row.get("main_ms") is not None:
        rows.append(row)
if len(rows) < 2:
    raise SystemExit("not enough rows")
rows.sort(key=lambda r: r["result_publish_timestamp_ns"])
span = (rows[-1]["result_publish_timestamp_ns"] - rows[0]["result_publish_timestamp_ns"]) / 1e9


def stats(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    return (round(values[len(values) // 2], 2),
            round(values[min(len(values) - 1, int(len(values) * 0.95))], 2),
            round(sum(values) / len(values), 2))


print("rows", len(rows), "span_s", round(span, 2),
      "result_fps", round(len(rows) / span, 2) if span else None)
print("main_ms", stats([r.get("main_ms") for r in rows]))
print("pipeline_ms", stats([r.get("pipeline_ms") for r in rows]))
print("decode_ms", stats([r.get("decode_ms") for r in rows]))
print("boxes", stats([r.get("boxes") for r in rows]),
      "counts", stats([r.get("main_count") for r in rows]))
print("last", rows[-1].get("frame_id"), rows[-1].get("boxes"),
      rows[-1].get("detection_counts"), rows[-1].get("traffic"))
