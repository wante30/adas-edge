"""Single-frame orchestrator; it has no knowledge of camera transport."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from adas_edge.contracts import Detection, Frame, PixelFormat
from adas_edge.decision.engine import ControlCommand, PerceptionState, RuleBasedDecisionEngine
from adas_edge.perception.base import Detector


@dataclass(frozen=True)
class PipelineResult:
    frame: Frame
    detections: tuple[Detection, ...]
    command: ControlCommand
    timing_ms: dict[str, float]
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.metadata(),
            "model": self.model_name,
            "detections": [detection.to_dict() for detection in self.detections],
            "command": {
                "action": self.command.action,
                "warning_flags": list(self.command.warning_flags),
                "confidence": self.command.confidence,
                "timestamp": self.command.timestamp,
            },
            "timing_ms": self.timing_ms,
        }


class AdasPipeline:
    def __init__(self, detector: Detector, decision_engine: RuleBasedDecisionEngine | None = None):
        self._detector = detector
        self._decision_engine = decision_engine or RuleBasedDecisionEngine()

    def process(self, frame: Frame) -> PipelineResult:
        started = perf_counter()
        preprocess_started = perf_counter()
        bgr = frame.as_bgr()
        prepared = Frame(
            camera_id=frame.camera_id,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            pixels=bgr,
            pixel_format=PixelFormat.BGR888,
        )
        preprocess_ms = (perf_counter() - preprocess_started) * 1000

        inference_started = perf_counter()
        detections = tuple(self._detector.detect(prepared))
        inference_ms = (perf_counter() - inference_started) * 1000

        postprocess_started = perf_counter()
        state = self._scene_state(detections, prepared)
        command = self._decision_engine.decide(state)
        postprocess_ms = (perf_counter() - postprocess_started) * 1000
        return PipelineResult(
            frame=prepared,
            detections=detections,
            command=command,
            timing_ms={
                "preprocess": round(preprocess_ms, 3),
                "inference": round(inference_ms, 3),
                "postprocess": round(postprocess_ms, 3),
                "end_to_end": round((perf_counter() - started) * 1000, 3),
            },
            model_name=self._detector.model_name,
        )

    @staticmethod
    def _scene_state(detections: tuple[Detection, ...], frame: Frame) -> PerceptionState:
        pedestrian_danger = any(
            detection.class_name == "person"
            and AdasPipeline._is_in_danger_zone(detection, frame)
            for detection in detections
        )
        obstacle = any(
            detection.class_name in {"car", "bus", "truck", "motorcycle", "bicycle"}
            and AdasPipeline._is_in_danger_zone(detection, frame)
            for detection in detections
        )
        traffic_light = "unknown" if any(
            detection.class_name == "traffic light" for detection in detections
        ) else None
        return PerceptionState(
            traffic_light=traffic_light,
            pedestrians_in_danger_zone=pedestrian_danger,
            obstacle_in_path=obstacle,
        )

    @staticmethod
    def _is_in_danger_zone(detection: Detection, frame: Frame) -> bool:
        x1, y1, x2, y2 = detection.bbox_xyxy
        center_x = (x1 + x2) / 2
        return frame.width * 0.2 <= center_x <= frame.width * 0.8 and y2 >= frame.height * 0.6
