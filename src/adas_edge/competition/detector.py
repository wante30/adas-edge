from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from adas_edge.contracts import Detection

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
)


class CompetitionDetector:
    """Detector adapter with injectable runner; RKNN is loaded only when requested."""
    def __init__(self, model_path: str | Path | None = None, input_size: int = 320, conf_threshold: float = 0.25, nms_threshold: float = 0.45, runner: Any = None):
        self.model_path = Path(model_path) if model_path else None
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self._runner = runner
        if self._runner is None:
            if self.model_path is None:
                raise ValueError("model_path is required without an injected runner")
            from rknnlite.api import RKNNLite
            self._runner = RKNNLite(verbose=False)
            if self._runner.load_rknn(str(self.model_path)) != 0 or self._runner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                raise RuntimeError(f"failed to initialize RKNN model: {self.model_path}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if hasattr(self._runner, "detect"):
            return self._sanitize(self._runner.detect(frame), frame.shape[1], frame.shape[0])
        if hasattr(self._runner, "inference"):
            outputs = self._runner.inference(inputs=[frame[None, ...]], data_format="nhwc")
            decoder = getattr(self._runner, "decode", None)
            rows = decoder(outputs) if decoder else []
            return self._sanitize(rows, frame.shape[1], frame.shape[0])
        return []

    @staticmethod
    def _sanitize(rows, width: int, height: int) -> list[Detection]:
        result: list[Detection] = []
        for row in rows:
            if isinstance(row, Detection):
                item = row
            else:
                bbox = row.get("bbox", row.get("bbox_xyxy")) if isinstance(row, dict) else getattr(row, "bbox_xyxy", None)
                if bbox is None: continue
                class_id = int(row.get("class_id", 0)) if isinstance(row, dict) else int(getattr(row, "class_id", 0))
                confidence = float(row.get("confidence", 0.0)) if isinstance(row, dict) else float(getattr(row, "confidence", 0.0))
                item = Detection(class_id, COCO_NAMES[class_id] if 0 <= class_id < len(COCO_NAMES) else f"class_{class_id}", confidence, tuple(float(x) for x in bbox))
            x1, y1, x2, y2 = item.bbox_xyxy
            if not all(math.isfinite(float(value)) for value in (x1, y1, x2, y2)) or x1 > x2 or y1 > y2: continue
            result.append(Detection(item.class_id, item.class_name, item.confidence, (max(0.0, min(width - 1.0, x1)), max(0.0, min(height - 1.0, y1)), max(0.0, min(width - 1.0, x2)), max(0.0, min(height - 1.0, y2)))))
        return result

    def close(self):
        if self._runner is not None and hasattr(self._runner, "release"):
            self._runner.release()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
