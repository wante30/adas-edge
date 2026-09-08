from datetime import datetime, timezone

import numpy as np

from adas_edge.contracts import Detection, Frame, PixelFormat
from adas_edge.pipeline import AdasPipeline


class FakeDetector:
    model_name = "fake-coco"

    def detect(self, frame: Frame) -> list[Detection]:
        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.95,
                bbox_xyxy=(40, 40, 80, 95),
            )
        ]


def test_pipeline_emits_json_serializable_result_and_stop_for_danger_pedestrian() -> None:
    frame = Frame(
        camera_id="front",
        frame_id=4,
        timestamp=datetime.now(timezone.utc),
        pixels=np.zeros((100, 100, 3), dtype=np.uint8),
        pixel_format=PixelFormat.BGR888,
    )

    result = AdasPipeline(detector=FakeDetector()).process(frame)

    payload = result.to_dict()
    assert payload["frame"]["camera_id"] == "front"
    assert payload["detections"][0]["class_name"] == "person"
    assert payload["command"]["action"] == "STOP"
    assert payload["timing_ms"]["end_to_end"] >= 0
