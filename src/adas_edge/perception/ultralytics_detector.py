from __future__ import annotations

from pathlib import Path

from adas_edge.contracts import Detection, Frame
from adas_edge.input_shape import to_ultralytics_imgsz, validate_model_shape
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
        input_size: int | str | tuple[int, int] = 640,
        device: str = "cpu",
        allowed_classes: set[str] | None = None,
        exact_input_canvas: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics to use the PC detector") from exc
        self._model = YOLO(str(model_path))
        self._confidence = confidence
        # Explicit (width, height); a bare int keeps its historical square meaning.
        self._input_width, self._input_height = validate_model_shape(input_size)
        self._input_size = to_ultralytics_imgsz(input_size)
        self._device = device
        self._exact_input_canvas = bool(exact_input_canvas)
        self.model_name = Path(model_path).name
        self._allowed_classes = set(ADAS_COCO_CLASSES if allowed_classes is None else allowed_classes)

    def detect(self, frame: Frame) -> list[Detection]:
        source = frame.as_bgr()
        scale, pad_x, pad_y = 1.0, 0, 0
        exact = getattr(self, "_exact_input_canvas", False)
        if exact:
            import cv2
            from tools.profile_rknn_single_stream import letterbox_bgr
            rgb, scale, pad_x, pad_y = letterbox_bgr(source, (self._input_width,self._input_height))
            # predict() accepts BGR NumPy images and performs its own RGB
            # conversion. Do not pass the RKNN RGB canvas directly to it.
            source = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = self._model.predict(
            source=source,
            conf=self._confidence,
            imgsz=self._input_size,
            device=self._device,
            verbose=False,
            rect=not exact,
        )[0]
        if result.boxes is None:
            return []
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(names[class_id])
            if class_name not in self._allowed_classes:
                continue
            x1, y1, x2, y2 = (float(item) for item in box.xyxy[0].tolist())
            if exact:
                h,w=frame.as_bgr().shape[:2]
                x1,x2=(max(0.,min(w,(v-pad_x)/scale)) for v in (x1,x2))
                y1,y2=(max(0.,min(h,(v-pad_y)/scale)) for v in (y1,y2))
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return detections
