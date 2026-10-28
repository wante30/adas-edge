"""Front-camera traffic-light specialist adapter.

The main detector is useful for broad scene objects, but traffic lights are
small enough that a COCO head is a poor sole source of signal candidates.  The
specialist uses the frozen 512x192 top-60% ROI contract used by the competition
model.  It supports both the PC ``best.pt`` model and the RK3566 hybrid RKNN
model, and maps every box back to full-frame coordinates before the common
colour/temporal filter consumes it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def signal_refresh_due(now_s: float, last_s: float | None, fps: float) -> bool:
    """Avoid accumulating one-frame drift at exact video-rate boundaries."""
    return (last_s is None or now_s < last_s or fps <= 0
            or now_s-last_s >= 1./fps-1e-6)


def horizontal_housing_hint(frame: np.ndarray, bbox: list[float]) -> str | None:
    """Recover a horizontal housing when a specialist boxes one lit bulb.

    An isolated bulb's square/tall box does not establish pedestrian type.
    Only a surrounding, connected horizontal dark housing produces a hint.
    This geometric display hint is not signal direction/ego-lane association.
    """
    x1, y1, x2, y2 = (round(v) for v in bbox)
    bw, bh = max(1, x2-x1), max(1, y2-y1)
    height, width = frame.shape[:2]
    rx1, ry1 = max(0, x1-3*bw), max(0, y1-bh)
    rx2, ry2 = min(width, x2+3*bw), min(height, y2+bh)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    # Bound the context operation independently of source resolution.
    scale = min(1., 256./max(roi.shape[:2]))
    small = cv2.resize(roi, None, fx=scale, fy=scale) if scale < 1 else roi
    dark = (cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2] < 120).astype(np.uint8)*255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    center = (((x1+x2)*.5-rx1)*scale, ((y1+y2)*.5-ry1)*scale)
    for contour in cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.pointPolygonTest(contour, center, False) < 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (w / max(1,h) >= 1.5 and w >= 1.5*bw*scale
                and .5*bh*scale <= h <= 2.5*bh*scale
                and w*h <= 12*bw*bh*scale*scale):
            return "VEHICLE_OR_HORIZONTAL"
    return None


def next_signal_roi(frame: np.ndarray, items: list[dict[str, Any]], roi_used: str, confidence: float = .05) -> str:
    """Keep a useful crop, or search the other crop on the next refresh."""
    from competition_ai.traffic_light import classify_traffic_light, TrafficLightState
    for item in items:
        state = classify_traffic_light(frame, item, min_confidence=confidence, require_lamp_evidence=True)
        x1,y1,x2,y2=item["bbox"]
        if state != TrafficLightState.UNKNOWN and ((x2-x1)/max(1.,y2-y1)>=1.25
                or horizontal_housing_hint(frame,item["bbox"]) is not None):
            return roi_used
    return "focus" if roi_used=="top60" else "top60"


class SignalCandidateCache:
    """Source-validated, bounded cache for the low-frequency signal branch.

    A box rejected on its inference frame cannot become a green signal when
    the camera later turns and that old rectangle covers leaves. Source time
    also expires boxes and clears them when a file is restarted.
    """

    def __init__(self, max_age_ms: float = 400.0, confidence: float = .05):
        self.max_age_ms = max(0., float(max_age_ms))
        self.confidence = float(confidence)
        self.reset()

    def reset(self) -> None:
        self.timestamp_ms: float | None = None
        self.items: list[dict[str, Any]] = []

    def store(self, frame: np.ndarray, items: list[dict[str, Any]], *, frame_id: int, timestamp_ms: float) -> None:
        from competition_ai.traffic_light import classify_traffic_light, TrafficLightState
        self.timestamp_ms = float(timestamp_ms)
        self.items = []
        for item in sorted(items, key=lambda d: float(d.get("confidence", 0.)), reverse=True)[:32]:
            state = classify_traffic_light(frame, item, min_confidence=self.confidence,
                                           require_lamp_evidence=True)
            self.items.append({**item, "lamp_evidence_valid": state != TrafficLightState.UNKNOWN,
                               "lamp_evidence_state": state.value, "signal_source_frame_id": int(frame_id),
                               "signal_source_timestamp_ms": float(timestamp_ms),
                               "signal_type_hint": horizontal_housing_hint(frame, item["bbox"])
                               if state != TrafficLightState.UNKNOWN else None})

    def current(self, timestamp_ms: float) -> list[dict[str, Any]]:
        if self.timestamp_ms is None:
            return []
        age = float(timestamp_ms) - self.timestamp_ms
        if age < 0 or age > self.max_age_ms:
            self.reset()
            return []
        return [{**item, "signal_detection_age_ms": round(age, 3)} for item in self.items]


def _letterbox_bgr(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, int, int]:
    """Return RGB NHWC-compatible input and the inverse mapping parameters."""

    width, height = size
    source_h, source_w = image.shape[:2]
    scale = min(width / max(1, source_w), height / max(1, source_h))
    resized_w, resized_h = round(source_w * scale), round(source_h * scale)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (width - resized_w) // 2, (height - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), scale, pad_x, pad_y


def _map_box(box: list[float], scale: float, pad_x: int, pad_y: int, crop_w: int, crop_h: int, roi_top: float, full_w: int, full_h: int, crop_origin: tuple[int, int] = (0, 0)) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    crop_x1 = max(0.0, min(float(crop_w), (x1 - pad_x) / max(scale, 1e-6)))
    crop_y1 = max(0.0, min(float(crop_h), (y1 - pad_y) / max(scale, 1e-6)))
    crop_x2 = max(0.0, min(float(crop_w), (x2 - pad_x) / max(scale, 1e-6)))
    crop_y2 = max(0.0, min(float(crop_h), (y2 - pad_y) / max(scale, 1e-6)))
    # The original top crop has origin (0,0). Focused search must restore
    # both offsets before clipping to the full source frame.
    del roi_top
    return [
        max(0.0, min(float(full_w), crop_x1 + crop_origin[0])),
        max(0.0, min(float(full_h), crop_y1 + crop_origin[1])),
        max(0.0, min(float(full_w), crop_x2 + crop_origin[0])),
        max(0.0, min(float(full_h), crop_y2 + crop_origin[1])),
    ]


class TrafficLightSpecialist:
    """Run the competition traffic-light specialist on a full BGR frame."""

    def __init__(
        self,
        model: str | Path,
        *,
        confidence: float = 0.05,
        iou_threshold: float = 0.70,
        roi_top: float = 0.60,
        input_size: tuple[int, int] = (512, 192),
        device: str = "cpu",
        roi_mode: str = "top60",
    ) -> None:
        self.model_path = Path(model)
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.roi_top = min(1.0, max(0.1, float(roi_top)))
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.backend = self.model_path.suffix.lower().lstrip(".") or "unknown"
        if roi_mode not in {"top60", "adaptive"}:
            raise ValueError("roi_mode must be top60 or adaptive")
        self.roi_mode = roi_mode
        self.reset_roi()
        self._runner: Any = None
        self._pc_detector: Any = None
        if self.backend == "pt":
            # The specialist has a single class named ``traffic light``.  The
            # normal Ultralytics adapter intentionally filters to ADAS classes,
            # so use it only after verifying its class name and retain the
            # adapter's timing-independent preprocessing contract.
            from adas_edge.perception.ultralytics_detector import UltralyticsDetector

            self._pc_detector = UltralyticsDetector(
                self.model_path,
                confidence=self.confidence,
                input_size=self.input_size,
                device=device,
                allowed_classes={"traffic light", "0"},
                exact_input_canvas=True,
            )
        elif self.backend == "rknn":
            from rknnlite.api import RKNNLite

            self._runner = RKNNLite(verbose=False)
            if self._runner.load_rknn(str(self.model_path)) != 0:
                raise RuntimeError(f"failed to load traffic-light specialist: {self.model_path}")
            if self._runner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                raise RuntimeError(f"failed to initialize traffic-light specialist: {self.model_path}")
        else:
            raise ValueError(f"traffic-light specialist must be .pt or .rknn, got {self.model_path}")

    def reset_roi(self) -> None:
        self._next_roi = "top60"
        self.last_roi = "top60"

    def detect(self, frame: np.ndarray, *, frame_id: int = 0) -> tuple[list[dict[str, Any]], float]:
        """Return full-frame candidate boxes and measured inference time."""

        if frame is None or frame.ndim != 3:
            return [], 0.0
        full_h, full_w = frame.shape[:2]
        roi_used = self._next_roi if self.roi_mode == "adaptive" else "top60"
        self.last_roi = roi_used
        if roi_used == "focus":
            x0, y0, x2, y2 = round(full_w*.20), round(full_h*.25), round(full_w*.80), round(full_h*.65)
        else:
            x0, y0, x2, y2 = 0, 0, full_w, max(1, int(round(full_h*self.roi_top)))
        crop = frame[y0:y2, x0:x2]
        started = time.perf_counter()
        if self._pc_detector is not None:
            from adas_edge.contracts import Frame, PixelFormat
            from datetime import datetime, timezone

            roi_frame = Frame(
                "front_signal_roi",
                int(frame_id),
                datetime.now(timezone.utc),
                crop,
                PixelFormat.BGR888,
            )
            raw = [item.to_dict() for item in self._pc_detector.detect(roi_frame)]
            # The trained specialist's dataset name is traffic light.  Keep a
            # defensive fallback for an exported one-class model whose names
            # are numeric.
            detections = [
                item
                for item in raw
                if str(item.get("class_name", "")).replace("_", " ").lower() in {"traffic light", "0"}
            ]
            mapped = []
            for item in detections:
                bbox = item.get("bbox") or item.get("bbox_xyxy")
                if bbox is None or len(bbox) != 4:
                    continue
                # Ultralytics returns coordinates in the source image passed to
                # predict(), which here is the cropped full-resolution ROI.
                x1, y1, x2, y2 = (float(value) for value in bbox)
                mapped.append({
                    "class_name": "traffic light",
                    "confidence": float(item.get("confidence", 0.0)),
                    "bbox": [
                        max(0.0, min(float(full_w), x1+x0)),
                        max(0.0, min(float(full_h), y1+y0)),
                        max(0.0, min(float(full_w), x2+x0)),
                        max(0.0, min(float(full_h), y2+y0)),
                    ],
                    "source": "TRAFFIC_LIGHT_SPECIALIST_PC",
                })
        else:
            from deployment.rknn.decode_yolo_output import decode_outputs

            canvas, scale, pad_x, pad_y = _letterbox_bgr(crop, self.input_size)
            outputs = self._runner.inference(inputs=[canvas[None, ...]], data_format="nhwc")
            raw = decode_outputs(outputs, self.confidence, self.iou_threshold)
            mapped = []
            for item in raw:
                mapped.append({
                    "class_name": "traffic light",
                    "confidence": float(item.get("confidence", 0.0)),
                    "bbox": _map_box(list(item["bbox"]), scale, pad_x, pad_y, crop.shape[1], crop.shape[0], self.roi_top, full_w, full_h, crop_origin=(x0,y0)),
                    "source": "TRAFFIC_LIGHT_SPECIALIST_RKNN",
                })
        if self.roi_mode == "adaptive":
            # Exactly one inference per refresh. Keep a useful ROI; otherwise
            # search the other window on the next scheduled refresh.
            self._next_roi = next_signal_roi(frame, mapped, roi_used, self.confidence)
        for item in mapped:
            item["specialist_roi"] = roi_used
        return mapped, round((time.perf_counter() - started) * 1000.0, 3)

    def close(self) -> None:
        if self._pc_detector is not None and hasattr(self._pc_detector, "close"):
            self._pc_detector.close()
        if self._runner is not None and hasattr(self._runner, "release"):
            self._runner.release()

