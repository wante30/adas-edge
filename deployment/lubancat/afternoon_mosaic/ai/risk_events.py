"""Canonical measured-risk events shared by presentation and control consumers."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import os
import time

BRAKE_TYPES = frozenset({"FRONT_COLLISION_RISK", "PEDESTRIAN_IN_PATH",
    "PEDESTRIAN_CROSSING_PATH", "TWO_WHEELER_PATH_CONFLICT", "VEHICLE_CUT_IN",
    "ONCOMING_PATH_CONFLICT", "SUSPECTED_BRAKING_WITH_CLOSING", "CROSSWALK_BRAKE_PULSE"})
SEVERITY_RANK = {"normal": 0, "info": 0, "warning": 1, "critical": 2}


class EventJournal:
    def __init__(self, path=None):
        path = path or os.environ.get("ADAS_EVENT_TIMELINE")
        self.fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644) if path else None

    def write(self, stage, **fields):
        if self.fd is not None:
            row = dict(stage=stage, logged_ns=time.monotonic_ns(), **fields)
            os.write(self.fd, (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))

    def close(self):
        if self.fd is not None:
            os.close(self.fd); self.fd = None


@dataclass(frozen=True)
class RiskObservation:
    timestamp_ns: int | None = None
    events: tuple = ()
    tracks: tuple = ()
    reason: str = "unified_events"
    source_id: str = "Q1"
    frame_id: int = -1
    road: dict = field(default_factory=dict)

    @property
    def pedestrian_hazard(self):
        return any(e["severity"] == "critical" and e.get("target_class") == "person" for e in self.events)

    @property
    def obstacle_hazard(self):
        return any(e["severity"] == "critical" and e.get("target_class") != "person" for e in self.events)


class EventStore:
    """One lifecycle, with bounded misses; UI consumers must never add a latch."""
    def __init__(self, journal=None, clear_ns=250_000_000, ttl_ns=400_000_000):
        self.journal = journal or EventJournal()
        self.clear_ns, self.ttl_ns = clear_ns, ttl_ns
        self.context = None
        self.active, self.misses, self.updates = {}, {}, []
        self.sequence = 0
        self.last_frame = None

    def _clear(self, key, now_ns, reason):
        event = dict(self.active.pop(key), lifecycle_state="CLEARED", risk_state="CLEARED",
                     valid=False, control_eligible=False, cleared_ns=now_ns, clear_reason=reason)
        self.misses.pop(key, None)
        self.updates.append(event)
        self.journal.write("event_transition", **event)

    def update(self, candidates, *, mode, epoch, bridge_id, model_id, frame_id,
               timestamp_ns, measurement_ns=None, now_ns=None, road_quadrant=0):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        measurement_ns = timestamp_ns if measurement_ns is None else measurement_ns
        forward_source = "Q%d" % (road_quadrant + 1)
        context = (bridge_id, mode, str(epoch), model_id, road_quadrant)
        frame_key = (context, frame_id, timestamp_ns)
        if self.last_frame == frame_key:
            return tuple(self.active.values())
        self.last_frame = frame_key
        self.updates = []
        if context != self.context:
            for key in list(self.active): self._clear(key, now_ns, "context_changed")
            self.context = context
        observed = set()
        for candidate in candidates:
            kind = candidate.get("event_type") or candidate.get("kind")
            severity = str(candidate.get("severity", "warning")).lower()
            if not kind or severity not in ("info", "warning", "critical"):
                continue
            if not 0 <= now_ns - measurement_ns <= self.ttl_ns:
                continue
            confidence=float(candidate.get("confidence",0.0))
            if not math.isfinite(confidence): continue
            source = candidate.get("source_id", "Q1")
            if source not in ("Q1","Q2","Q3","Q4") or (mode=="ROAD_Q1" and source!=forward_source): continue
            track_id = int(candidate.get("track_id") or 0)
            bbox = candidate.get("bbox")
            if bbox is not None:
                if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox): continue
                x1, y1, x2, y2 = map(float, bbox)
                if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 360): continue
                if mode == "ROAD_Q1": bbox = [round(2*v, 2) for v in (x1,y1,x2,y2)]
                else:
                    index = int(source[-1]) - 1 if source in ("Q1","Q2","Q3","Q4") else 0
                    ox, oy = (index % 2) * 640, (index // 2) * 360
                    bbox = [round(x1+ox,2),round(y1+oy,2),round(x2+ox,2),round(y2+oy,2)]
            key = (source, track_id, kind)
            previous = self.active.get(key)
            if previous is None: self.sequence += 1
            event = dict(event_id=previous["event_id"] if previous else
                         f"{bridge_id}:{epoch}:{self.sequence}", kind=kind, event_type=kind,
                         severity=severity, risk_state=severity.upper(), lifecycle_state="ACTIVE",
                         source_id=source, inference_mode=mode, mode_epoch=str(epoch),
                         frame_id=int(frame_id), timestamp_ns=int(measurement_ns), track_id=track_id,
                         target_class=candidate.get("target_class") or candidate.get("class_name", ""),
                         bbox=bbox, coordinate_space="ROAD_VIEW_1280x720" if mode=="ROAD_Q1" else "MOSAIC_1280x720",
                         confidence=round(float(candidate.get("confidence", 0.0)),3),
                         evidence=candidate.get("evidence") or candidate.get("source", "MEASURED_DETECTIONS"),
                         path_valid=bool(candidate.get("path_valid", True)),
                         control_eligible=bool(severity=="critical" and source==forward_source and track_id>0
                             and bbox and candidate.get("path_valid",True) and kind in BRAKE_TYPES and candidate.get("control_eligible", True)),
                         valid=True, expires_at_ns=int(measurement_ns+self.ttl_ns),
                         confirmed_ns=previous["confirmed_ns"] if previous else now_ns,
                         synthetic=bool(candidate.get("synthetic", False)))
            self.active[key] = event; self.misses[key] = 0; observed.add(key)
            if previous is None or previous["severity"] != severity:
                self.updates.append(event); self.journal.write("event_transition", **event)
        for key in list(self.active):
            if key in observed: continue
            self.misses[key] = self.misses.get(key,0)+1
            event = self.active[key]
            self.active[key] = dict(event, control_eligible=False)
            if self.misses[key]>=2 or now_ns-event["timestamp_ns"]>=self.clear_ns or now_ns>event["expires_at_ns"]:
                self._clear(key, now_ns, "risk_no_longer_observed")
        return tuple(self.active.values())
