from __future__ import annotations

from dataclasses import dataclass

from adas_edge.contracts import Detection

from .pixel import payload_to_bgr
from .policy import CompetitionPolicy, LaneState, PolicyResult, SceneInput, TrafficLightState
from .protocol import DetectionRecord, FrameDescriptor, ResultDescriptor


@dataclass(frozen=True)
class PipelineOutput:
    descriptor: ResultDescriptor
    detections: tuple[Detection, ...]
    policy: PolicyResult


class CompetitionPipeline:
    def __init__(self, detector, policy: CompetitionPolicy | None = None) -> None:
        self.detector = detector
        self.policy = policy or CompetitionPolicy()

    def process_descriptor(self, frame_descriptor: FrameDescriptor, scene_context: SceneInput | None = None) -> ResultDescriptor:
        frame = payload_to_bgr(frame_descriptor.payload, frame_descriptor.width, frame_descriptor.height, frame_descriptor.pixel_format, frame_descriptor.stride)
        detections = tuple(self._clip_detection(item, frame_descriptor.width, frame_descriptor.height) for item in self.detector.detect(frame))
        detections = tuple(item for item in detections if item is not None)
        scene = scene_context or SceneInput()
        policy = self.policy.decide(scene)
        records = tuple(item for item in (DetectionRecord.from_detection(d, frame_descriptor.width, frame_descriptor.height) for d in detections) if item is not None)
        return ResultDescriptor(frame_descriptor.stream_id, frame_descriptor.frame_id, frame_descriptor.timestamp_ns, len(records), scene.traffic_light_state, scene.lane_state, policy.action, policy.warning_flags, records)

    @staticmethod
    def _clip_detection(detection: Detection, width: int, height: int):
        import math
        x1, y1, x2, y2 = detection.bbox_xyxy
        values = [float(x1), float(y1), float(x2), float(y2)]
        if not all(math.isfinite(value) for value in values):
            return None
        x1, y1, x2, y2 = values
        if x1 > x2 or y1 > y2:
            return None
        x1, x2 = max(0.0, min(width - 1.0, x1)), max(0.0, min(width - 1.0, x2))
        y1, y2 = max(0.0, min(height - 1.0, y1)), max(0.0, min(height - 1.0, y2))
        return Detection(detection.class_id, detection.class_name, detection.confidence, (x1, y1, x2, y2))
