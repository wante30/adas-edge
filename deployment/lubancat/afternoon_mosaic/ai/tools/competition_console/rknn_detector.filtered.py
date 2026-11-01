"""Adapt the RKNN detector to the native console's detection interface."""
from __future__ import annotations

from pathlib import Path

from competition_ai.detector import COMPETITION_CLASSES


class RknnConsoleDetector:
    def __init__(self, model_path: str | Path, input_size, *, specialist=False):
        from competition_ai.detector import CompetitionDetector

        self.specialist = specialist
        self.source = "TL_SPECIALIST_RKNN" if specialist else "MAIN_RKNN"
        self.frame_id = 0
        self.detector = CompetitionDetector(
            model_path,
            input_size=input_size,
            conf_threshold=0.05 if specialist else 0.25,
            nms_threshold=0.70 if specialist else 0.45,
        )

    def detect(self, frame):
        result = self.detector.infer(
            frame, stream_id=0, frame_id=self.frame_id, timestamp_ns=None
        )
        self.frame_id += 1
        items = []
        for detection in result.detections:
            class_name = "traffic light" if self.specialist else detection.class_name
            # Normalize BDD100K/legacy names to the UI vocabulary.
            if class_name in ("pedestrian", "rider"):
                class_name = "person"
            elif class_name in ("motor", "motorcycle"):
                class_name = "motorcycle"
            elif class_name in ("bike", "bicycle"):
                class_name = "bicycle"
            elif class_name == "traffic sign":
                continue
            if not self.specialist:
                # The main BDD model has weak traffic-light/motor/bike
                # precision. Traffic lights are handled by the specialist.
                if class_name == "traffic light":
                    continue
                if class_name in ("motorcycle", "bicycle") and detection.confidence < 0.85:
                    continue
            if class_name not in COMPETITION_CLASSES:
                continue
            items.append({
                "class_name": class_name,
                "confidence": detection.confidence,
                "bbox": [detection.x1, detection.y1, detection.x2, detection.y2],
                "source": self.source,
            })
        return items, result.inference_ms

    def close(self):
        self.detector.close()
