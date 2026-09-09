"""Deprecated compatibility facade for the canonical ABI."""
from adas_edge.competition.protocol import (
    DETECTION, FRAME_HEADER, FRAME_MAGIC, RESULT_HEADER, RESULT_MAGIC, VERSION,
    DetectionRecord as _DetectionRecord, FrameDescriptor, ProtocolError,
    ResultDescriptor as _ResultDescriptor,
)


class DetectionRecord(_DetectionRecord):
    def __init__(self, class_id, confidence, x1, y1, x2, y2):
        value = round(float(confidence) * 32767) if isinstance(confidence, float) else int(confidence)
        super().__init__(int(class_id), value, int(x1), int(y1), int(x2), int(y2))


class ResultDescriptor(_ResultDescriptor):
    """Accept the former positional constructor while serializing canonical bytes."""
    def __init__(self, stream_id, frame_id, timestamp_ns, detections=(), traffic_light_state=0, lane_state=0, adas_action=0, warning_flags=0):
        super().__init__(stream_id, frame_id, timestamp_ns, len(detections), traffic_light_state, lane_state, adas_action, warning_flags, tuple(detections))

    @classmethod
    def deserialize(cls, packet):
        result = _ResultDescriptor.deserialize(packet)
        detections = tuple(DetectionRecord(item.class_id, item.confidence_q15, item.x1, item.y1, item.x2, item.y2) for item in result.detections)
        return cls(result.stream_id, result.frame_id, result.timestamp_ns, detections, result.traffic_light_state, result.lane_state, result.adas_action, result.warning_flags)
