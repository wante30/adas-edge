from __future__ import annotations

from pathlib import Path

from adas_edge.contracts import Detection, Frame
from adas_edge.perception.base import Detector


ADAS_COCO_CLASSES = {
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "traffic light",
    "stop sign",
}


class UltralyticsDetector(Detector):
    """Ultralytics adapter; the rest of the repository never imports its API."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        input_size: int = 640,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics to use the PC detector") from exc
        self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._input_size = input_size
        self._device = device
        self.model_name = Path(model_path).name

    def detect(self, frame: Frame) -> list[Detection]:
        result = self._model.predict(
            source=frame.as_bgr(),
            conf=self._confidence,
            imgsz=self._input_size,
            device=self._device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(names[class_id])
            if class_name not in ADAS_COCO_CLASSES:
                continue
            x1, y1, x2, y2 = (float(item) for item in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return detections
