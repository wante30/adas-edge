"""Runtime traffic-light state estimation for the front-camera path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from competition_ai.traffic_light import TrafficLightState, classify_traffic_light


@dataclass(frozen=True)
class TrafficLightEstimate:
    state: str = "UNKNOWN"
    confidence: float = 0.0
    bbox: list[float] | None = None
    raw_state: str = "UNKNOWN"
    candidate_count: int = 0
    signal_type: str = "UNKNOWN"
    pedestrian_state: str = "UNKNOWN"
    vehicle_candidate_count: int = 0
    pedestrian_candidate_count: int = 0
    decision: str = "UNKNOWN"
    association: str = "UNVERIFIED"
    association_confidence: float = 0.0

    @property
    def decision_state(self) -> str:
        return self.decision

    def as_metadata(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "raw_state": self.raw_state,
            "candidate_count": int(self.candidate_count),
            "signal_type": self.signal_type,
            "pedestrian_state": self.pedestrian_state,
            "vehicle_candidate_count": int(self.vehicle_candidate_count),
            "pedestrian_candidate_count": int(self.pedestrian_candidate_count),
            "decision_state": self.decision_state,
            "association": self.association,
            "association_confidence": round(float(self.association_confidence), 4),
            "source": "DETECTOR_ROI_HSV",
            "validation": "EXPERIMENTAL",
        }


class TrafficLightTemporalFilter:
    """Turn noisy frame-wise signal colors into a conservative stable state.

    Detection and color classification are deliberately separate. UNKNOWN is
    retained until a color is seen for ``confirm_frames`` updates, and a stable
    color is held briefly through detector gaps so the HMI does not flicker.
    """

    def __init__(self, min_confidence: float = 0.25, confirm_frames: int = 2, hold_frames: int = 4, pedestrian_hold_frames: int | None = 4, ego_x_min: float = 0.20, ego_x_max: float = 0.76, horizon_y_max: float = 0.78, association_mode: str = "unverified", specialist_min_confidence: float | None = .05, require_lamp_evidence: bool = True):
        self.min_confidence = float(min_confidence)
        self.specialist_min_confidence = float(min_confidence if specialist_min_confidence is None else specialist_min_confidence)
        self.require_lamp_evidence = bool(require_lamp_evidence)
        self.confirm_frames = max(1, int(confirm_frames))
        self.hold_frames = max(0, int(hold_frames))
        self.pedestrian_hold_frames = max(0, int(self.hold_frames if pedestrian_hold_frames is None else pedestrian_hold_frames))
        self.ego_x_min = float(ego_x_min)
        self.ego_x_max = float(ego_x_max)
        self.horizon_y_max = float(horizon_y_max)
        if association_mode not in {"unverified", "geometry-experimental"}:
            raise ValueError("association_mode must be 'unverified' or 'geometry-experimental'")
        self.association_mode = association_mode
        self._candidate = TrafficLightState.UNKNOWN
        self._candidate_count = 0
        self._stable = TrafficLightState.UNKNOWN
        self._hold_count = 0
        self._ped_candidate = TrafficLightState.UNKNOWN
        self._ped_candidate_count = 0
        self._ped_stable = TrafficLightState.UNKNOWN
        self._ped_hold_count = 0
        self._association_count = 0
        self._association_stable = False
        self._association_hold_count = 0
        self._last = TrafficLightEstimate()

    def reset(self) -> None:
        self._candidate = TrafficLightState.UNKNOWN
        self._candidate_count = 0
        self._stable = TrafficLightState.UNKNOWN
        self._hold_count = 0
        self._ped_candidate = TrafficLightState.UNKNOWN
        self._ped_candidate_count = 0
        self._ped_stable = TrafficLightState.UNKNOWN
        self._ped_hold_count = 0
        self._association_count = 0
        self._association_stable = False
        self._association_hold_count = 0
        self._last = TrafficLightEstimate()

    @staticmethod
    def _traffic_items(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            item
            for item in detections
            if str(item.get("class_name", "")).replace("_", " ").lower() == "traffic light"
        ]

    @staticmethod
    def _signal_type(bbox: list[float]) -> str:
        """Classify the detector box shape for display/association hints.

        This is deliberately a geometry hint, not a replacement for a signal
        classifier. The competition clip contains horizontal vehicle signals
        and a tall pedestrian signal; retaining the distinction prevents the
        latter from being silently presented as an ego-lane vehicle light.
        """

        x1, y1, x2, y2 = (float(value) for value in bbox)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        ratio = width / height
        if ratio >= 1.25:
            return "VEHICLE_OR_HORIZONTAL"
        if ratio <= 0.80:
            return "PEDESTRIAN_OR_VERTICAL"
        return "UNKNOWN"

    @staticmethod
    def _geometry_associated(bbox: list[float], frame: np.ndarray, lane_metadata: dict[str, Any] | None, signal_type: str) -> bool:
        """Experimental ego-lane gate for a horizontal signal candidate.

        The lane runtime is a display candidate rather than a calibrated road
        model, so this gate is opt-in and remains explicitly experimental. It
        requires both lane boundaries, a forward-facing horizontal signal, and
        a candidate in the upper central corridor. It never runs by default.
        """

        if signal_type != "VEHICLE_OR_HORIZONTAL" or not lane_metadata:
            return False
        left = lane_metadata.get("left") or {}
        right = lane_metadata.get("right") or {}
        left_points = left.get("points") or []
        right_points = right.get("points") or []
        if not left_points or not right_points:
            return False
        x1, y1, x2, y2 = (float(value) for value in bbox)
        height, width = frame.shape[:2]
        center_x = ((x1 + x2) * 0.5) / max(1.0, float(width))
        center_y = ((y1 + y2) * 0.5) / max(1.0, float(height))
        if not 0.25 <= center_x <= 0.75 or center_y > 0.55:
            return False
        try:
            left_x = float(left_points[-1]["x"])
            right_x = float(right_points[-1]["x"])
        except (KeyError, TypeError, ValueError):
            return False
        if left_x > right_x:
            left_x, right_x = right_x, left_x
        lateral = (center_x - 0.5) * 10.0
        margin = max(1.25, abs(right_x - left_x) * 0.65)
        return left_x - margin <= lateral <= right_x + margin

    def update(self, frame: np.ndarray, detections: list[dict[str, Any]], lane_metadata: dict[str, Any] | None = None) -> TrafficLightEstimate:
        candidates = self._traffic_items(detections)
        ranked: list[tuple[int, float, TrafficLightState, list[float], str]] = []
        pedestrian_ranked: list[tuple[float, TrafficLightState, list[float]]] = []
        for item in candidates:
            confidence = float(item.get("confidence", 0.0))
            is_specialist = str(item.get("source", "")) in {"TRAFFIC_LIGHT_SPECIALIST_PC", "TRAFFIC_LIGHT_SPECIALIST_RKNN"}
            threshold = self.specialist_min_confidence if is_specialist else self.min_confidence
            if confidence < threshold:
                continue
            if item.get("lamp_evidence_valid") is False:
                continue
            if float(item.get("signal_detection_age_ms", 0.)) > 400.:
                continue
            bbox = item.get("bbox") or item.get("bbox_xyxy")
            if bbox is None or len(bbox) != 4:
                continue
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = (float(value) for value in bbox)
            center_x = ((x1 + x2) * 0.5) / max(1.0, width)
            center_y = ((y1 + y2) * 0.5) / max(1.0, height)
            # This only limits displayed candidates. It cannot establish
            # direction or which lane a signal controls; association below
            # stays a separate opt-in gate.
            if not self.ego_x_min <= center_x <= self.ego_x_max or center_y > self.horizon_y_max:
                continue
            if is_specialist and item.get("lamp_evidence_valid") is True:
                # Keep the bounded observation from its inference image;
                # reclassifying an old rectangle on a new image measures a
                # different patch as the camera turns.
                try:
                    state = TrafficLightState(item.get("lamp_evidence_state", "UNKNOWN"))
                except ValueError:
                    state = TrafficLightState.UNKNOWN
            else:
                state = classify_traffic_light(frame, item, min_confidence=threshold,
                                               require_lamp_evidence=self.require_lamp_evidence)
            if state != TrafficLightState.UNKNOWN:
                bbox = item.get("bbox") or item.get("bbox_xyxy")
                if bbox is not None:
                    bbox_values = [float(value) for value in bbox]
                    signal_type = self._signal_type(bbox_values)
                    if item.get("lamp_evidence_valid") is True and item.get("signal_type_hint") == "VEHICLE_OR_HORIZONTAL":
                        signal_type = "VEHICLE_OR_HORIZONTAL"
                    if is_specialist and self.require_lamp_evidence and signal_type == "UNKNOWN":
                        # A single colour patch on a round roadside sign is
                        # not a vehicle signal. Keep the detector box as an
                        # unknown candidate until housing context resolves it.
                        continue
                    # Prefer a horizontal/vehicle-shaped candidate if both a
                    # pedestrian and vehicle signal are visible in one frame;
                    # confidence remains the secondary ranking criterion.
                    type_priority = 2 if signal_type == "VEHICLE_OR_HORIZONTAL" else 1 if signal_type == "UNKNOWN" else 0
                    if signal_type == "PEDESTRIAN_OR_VERTICAL":
                        pedestrian_ranked.append((confidence, state, bbox_values))
                    else:
                        ranked.append((type_priority, confidence, state, bbox_values, signal_type))

        if pedestrian_ranked:
            pedestrian_raw = max(pedestrian_ranked, key=lambda item: item[0])[1]
            if pedestrian_raw == self._ped_candidate:
                self._ped_candidate_count += 1
            else:
                self._ped_candidate = pedestrian_raw
                self._ped_candidate_count = 1
            if self._ped_candidate_count >= self.confirm_frames:
                self._ped_stable = pedestrian_raw
            self._ped_hold_count = 0
        else:
            pedestrian_raw = TrafficLightState.UNKNOWN
            if self._ped_stable != TrafficLightState.UNKNOWN and self._ped_hold_count < self.pedestrian_hold_frames:
                self._ped_hold_count += 1
            else:
                self._ped_stable = TrafficLightState.UNKNOWN
                self._ped_candidate = TrafficLightState.UNKNOWN
                self._ped_candidate_count = 0
        pedestrian_state = self._ped_stable.value if self._ped_stable != TrafficLightState.UNKNOWN else "UNKNOWN"
        vehicle_candidate_count = len(ranked)
        pedestrian_candidate_count = len(pedestrian_ranked)

        if ranked:
            _, confidence, raw_state, bbox, signal_type = max(ranked, key=lambda item: (item[0], item[1]))
            if raw_state == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = raw_state
                self._candidate_count = 1
            if self._candidate_count >= self.confirm_frames:
                self._stable = raw_state
            self._hold_count = 0
        else:
            raw_state = TrafficLightState.UNKNOWN
            confidence = 0.0
            bbox = None
            signal_type = "UNKNOWN"
            if self._stable != TrafficLightState.UNKNOWN and self._hold_count < self.hold_frames:
                self._hold_count += 1
            else:
                self._stable = TrafficLightState.UNKNOWN
                self._candidate = TrafficLightState.UNKNOWN
                self._candidate_count = 0

        association_bbox = None
        association_type = "UNKNOWN"
        if self._stable != TrafficLightState.UNKNOWN:
            if ranked and raw_state == self._stable:
                association_bbox = bbox
                association_type = signal_type
            elif not ranked:
                association_bbox = self._last.bbox
                association_type = self._last.signal_type
        associated_now = (
            self.association_mode == "geometry-experimental"
            and association_bbox is not None
            and self._geometry_associated(association_bbox, frame, lane_metadata, association_type)
        )
        if associated_now:
            self._association_count += 1
            self._association_hold_count = 0
            if self._association_count >= self.confirm_frames:
                self._association_stable = True
        elif self._association_stable and not ranked and self._association_hold_count < self.hold_frames:
            self._association_hold_count += 1
        else:
            self._association_count = 0
            self._association_stable = False
            self._association_hold_count = 0
        association = "GEOMETRIC_EXPERIMENTAL" if self._association_stable else "UNVERIFIED"
        decision = self._stable.value if self._association_stable else "UNKNOWN"
        association_confidence = 1.0 if self._association_stable else 0.0

        if self._stable == TrafficLightState.UNKNOWN:
            estimate = TrafficLightEstimate(
                state="UNKNOWN",
                confidence=confidence if ranked else 0.0,
                bbox=bbox,
                raw_state=raw_state.value,
                candidate_count=len(candidates),
                signal_type=signal_type,
                pedestrian_state=pedestrian_state,
                vehicle_candidate_count=vehicle_candidate_count,
                pedestrian_candidate_count=pedestrian_candidate_count,
                decision=decision,
                association=association,
                association_confidence=association_confidence,
            )
        else:
            estimate = TrafficLightEstimate(
                state=self._stable.value,
                confidence=confidence if ranked else self._last.confidence,
                bbox=bbox if bbox is not None else self._last.bbox,
                raw_state=raw_state.value,
                candidate_count=len(candidates),
                signal_type=signal_type if bbox is not None else self._last.signal_type,
                pedestrian_state=pedestrian_state,
                vehicle_candidate_count=vehicle_candidate_count,
                pedestrian_candidate_count=pedestrian_candidate_count,
                decision=decision,
                association=association,
                association_confidence=association_confidence,
            )
        self._last = estimate
        return estimate

