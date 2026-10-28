import json
from ai_bridge import build_dynamic_status

now = 10_000_000_000
status = build_dynamic_status(
    source="LOCAL_REPLAY", frame_id=100, timestamp_ns=9_900_000_000, now_ns=now,
    hud={"traffic": "RED", "traffic_evidence": "ONE_DETECTED_RED_SIGNAL"},
    navigation={"valid": True, "direction": "forward", "distance_m": 100.0,
                "road_name": "测试路", "source_frame_id": 95, "timestamp_ns": 9_500_000_000,
                "ocr_timestamp_ns": 9_800_000_000, "reason": "Q2_FIXTURE"},
    hazard_timestamp_ns=9_800_000_000,
    hazards=[{"kind": "PEDESTRIAN_IN_PATH", "source": "IMAGE_SPACE_PATH_PRESENCE"}],
    detections=[], signal_age_ms=100.0, main_age_ms=100.0, specialist_age_ms=100.0,
    navigation_ttl_ms=5000, source_stopped=False, source_available=True,
)
print(json.dumps(status, ensure_ascii=False, indent=2))
