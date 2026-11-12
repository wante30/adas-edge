"""Adapt the RKNN detector to the native console's detection interface."""
from __future__ import annotations

from pathlib import Path

from competition_ai.detector import COMPETITION_CLASSES


class RknnConsoleDetector:
    def __init__(self, model_path: str | Path, input_size, *, specialist=False,
                 class_names=None, class_profile=None, include_traffic_lights=False,
                 confidence_threshold=None, nms_threshold=None, backend=None, task=None):
        from competition_ai.detector import CompetitionDetector

        self.specialist = bool(specialist)
        self.include_traffic_lights = bool(include_traffic_lights)
        self.source = "TL_SPECIALIST_RKNN" if self.specialist else "MAIN_RKNN"
        self.frame_id = 0
        self.detector = CompetitionDetector(
            model_path,
            input_size=input_size,
            conf_threshold=(0.05 if self.specialist else 0.25) if confidence_threshold is None else float(confidence_threshold),
            nms_threshold=(0.70 if self.specialist else 0.45) if nms_threshold is None else float(nms_threshold),
            class_names=class_names,
            class_profile=class_profile,
            use_rga_preprocess=not self.specialist,
        )
        # The native bridge has profile metadata for CAPI/MTL backends.  This
        # Lite adapter intentionally keeps the existing raw-head path; retain
        # the arguments so model-profile construction remains compatible.
        self.backend = backend or "rknn_lite"
        self.task = task or "detect"
        self.last_timing = None

    def detect(self, frame):
        result = self.detector.infer(
            frame, stream_id=0, frame_id=self.frame_id, timestamp_ns=None
        )
        self.frame_id += 1
        self.last_timing = dict(getattr(self.detector, "last_timing", None) or {})
        items = []
        for detection in result.detections:
            class_name = "traffic light" if self.specialist else detection.class_name
            if class_name == "pedestrian":
                class_name = "person"
            elif class_name in ("motor", "motorcycle"):
                class_name = "motorcycle"
            elif class_name in ("bike", "bicycle"):
                class_name = "bicycle"
            elif class_name == "traffic sign":
                continue
            if not self.specialist:
                if class_name == "traffic light" and not self.include_traffic_lights:
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
