from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\ai_bridge.py')
s=p.read_text(encoding='utf-8')
old='''    nav_fresh = age_ms is not None and age_ms < navigation_ttl_ms
    nav_valid = bool(navigation.get("valid")) and nav_identity_valid and nav_fresh and not source_stopped
    if source_stopped or (navigation.get("valid") and nav_identity_valid and not nav_fresh):
        nav_state = "STOPPED"
    elif nav_valid and navigation.get("direction") in NAV_DIRECTIONS:
        nav_state = "VALID"
    else:
        nav_state = "UNREADABLE"
    nav = {
        "state": nav_state,
        "direction": navigation.get("direction", "none") if nav_state == "VALID" else "none",
        "distance_m": navigation.get("distance_m") if nav_state == "VALID" else None,
        "road_name": str(navigation.get("road_name") or "") if nav_state == "VALID" else "",
        "source_frame_id": nav_source,
        "age_ms": age_ms,
    }'''
new='''    nav_fresh = age_ms is not None and age_ms < navigation_ttl_ms
    ocr_timestamp = navigation.get("ocr_timestamp_ns")
    ocr_age_ms = ((now_ns - ocr_timestamp) / 1_000_000
                  if isinstance(ocr_timestamp, int) and 0 <= ocr_timestamp <= now_ns else None)
    ocr_fresh = ocr_age_ms is not None and ocr_age_ms < navigation_ttl_ms
    nav_valid = (bool(navigation.get("valid")) and nav_identity_valid
                 and nav_fresh and ocr_fresh and not source_stopped)
    if source_stopped:
        nav_state = "STOPPED"
    elif navigation.get("valid") and nav_identity_valid and (not nav_fresh or not ocr_fresh):
        nav_state = "STALE"
    elif nav_valid and navigation.get("direction") in NAV_DIRECTIONS:
        nav_state = "VALID"
    else:
        nav_state = "UNREADABLE"
    nav = {
        "state": nav_state,
        "direction": navigation.get("direction", "none") if nav_state == "VALID" else "none",
        "distance_m": navigation.get("distance_m") if nav_state == "VALID" else None,
        "road_name": str(navigation.get("road_name") or "") if nav_state == "VALID" else "",
        "source_frame_id": nav_source,
        "age_ms": age_ms,
        "ocr_timestamp_ns": ocr_timestamp,
        "ocr_age_ms": ocr_age_ms,
        "reason": navigation.get("reason", ""),
        "valid": nav_state == "VALID",
    }'''
if old not in s: raise SystemExit('nav block not found')
s=s.replace(old,new)
old2='''    hazard_kinds = []
    if (not source_stopped and source_available and hazard_age_ms is not None
            and hazard_age_ms < DYNAMIC_RESULT_TTL_MS):
        hazard_kinds = list(dict.fromkeys(
            str(item.get("kind")) for item in hazards
            if isinstance(item, dict) and isinstance(item.get("kind"), str)
        ))'''
new2='''    hazard_kinds = []
    hazard_details = []
    if (not source_stopped and source_available and hazard_age_ms is not None
            and hazard_age_ms < DYNAMIC_RESULT_TTL_MS):
        for item in hazards:
            if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
                continue
            kind = item["kind"]
            hazard_kinds.append(kind)
            hazard_details.append({
                "event_type": kind,
                "source_id": source,
                "source_frame_id": int(frame_id),
                "timestamp": int(hazard_timestamp_ns),
                "severity": "warning",
                "valid": True,
                "ttl_ms": DYNAMIC_RESULT_TTL_MS,
                "evidence": item.get("source") or "IMAGE_PERCEPTION",
            })
    if (not source_stopped and source_available and traffic == "RED"
            and signal_age_ms is not None and signal_age_ms < 2500):
        hazard_kinds.append("RED_SIGNAL_CONFIRMED")
        hazard_details.append({
            "event_type": "RED_SIGNAL_CONFIRMED",
            "source_id": source,
            "source_frame_id": int(frame_id),
            "timestamp": int(timestamp_ns),
            "severity": "info",
            "valid": True,
            "ttl_ms": 2500,
            "evidence": traffic_reason,
        })
    hazard_kinds = list(dict.fromkeys(hazard_kinds))'''
if old2 not in s: raise SystemExit('hazard block not found')
s=s.replace(old2,new2)
s=s.replace('''        "hazards": hazard_kinds,
        "hazard_age_ms": hazard_age_ms,''','''        "hazards": hazard_kinds,
        "hazard_details": hazard_details,
        "hazard_age_ms": hazard_age_ms,''')
p.write_text(s,encoding='utf-8')
print('status patched')
