"""PC adapter for the official Autoware MobileNetV2 traffic-light model.

The adapter is deliberately independent from detection and tracking.  It
classifies an already-cropped BGR ROI and keeps the official comma-separated
label in ``evidence`` so compound labels are never silently rewritten as one
arrow direction.
"""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from .multi_signal import Attributes


DEFAULT_LABELS = (
    "green", "left,red", "left,red,straight", "red", "red,right",
    "red,straight", "unknown", "yellow", "red,up_left",
    "red,right,straight", "red,up_right",
)
COLOR_NAMES = {"red": "RED", "yellow": "YELLOW", "green": "GREEN"}
SHAPE_NAMES = {
    "left": "LEFT_ARROW", "straight": "STRAIGHT_ARROW", "right": "RIGHT_ARROW",
    "up_left": "LEFT_ARROW", "up_right": "RIGHT_ARROW",
}


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / np.maximum(probabilities.sum(), 1e-12)


def _probabilities(values: np.ndarray) -> tuple[np.ndarray, str]:
    """Interpret the official ONNX output without applying softmax twice.

    The released MobileNetV2 file emits a probability vector (non-negative,
    sum approximately one), while exported variants may emit logits.  Keep
    both cases explicit so confidence values remain meaningful.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    total = float(values.sum())
    if np.all(np.isfinite(values)) and np.all(values >= 0.0) and 0.98 <= total <= 1.02:
        return values / max(total, 1e-12), "probabilities"
    return _softmax(values), "logits"


def parse_label(label: str) -> tuple[str, str, list[str], str]:
    """Map an official label to color, safe single shape and directions.

    Official labels put color and direction tokens in one comma-separated
    string (for example ``left,red,straight``).  A multi-direction label has
    no single ``shape`` in the existing interface, so it remains UNKNOWN and
    carries all directions in evidence.
    """
    tokens = [token.strip().lower() for token in str(label).split(",") if token.strip()]
    if not tokens or tokens == ["unknown"]:
        return "UNKNOWN", "UNKNOWN", [], "AUTOWARE_UNKNOWN_LABEL"
    color_token = next((token for token in tokens if token in COLOR_NAMES), None)
    color = COLOR_NAMES.get(color_token, "UNKNOWN")
    directions = [SHAPE_NAMES[token] for token in tokens if token in SHAPE_NAMES]
    if not directions:
        shape = "ROUND" if color != "UNKNOWN" else "UNKNOWN"
        reason = "AUTOWARE_COLOR_ONLY"
    elif len(directions) == 1:
        shape = directions[0]
        reason = "AUTOWARE_SINGLE_DIRECTION"
    else:
        shape = "UNKNOWN"
        reason = "AUTOWARE_COMBINATION"
    return color, shape, directions, reason


class AutowareMobileNetV2Backend:
    """Run a batch-1 Autoware classifier ONNX model with official preprocessing."""

    def __init__(self, model_path: str | Path, labels_path: str | Path | None = None,
                 *, input_size: tuple[int, int] = (224, 224), providers=None):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if labels_path is None:
            labels = list(DEFAULT_LABELS)
        else:
            labels = [line.strip() for line in Path(labels_path).read_text(encoding="utf-8").splitlines()
                      if line.strip()]
        if not labels:
            raise ValueError("Autoware label list is empty")
        self.labels = labels
        self.input_size = tuple(int(value) for value in input_size)
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised on dependency-free hosts
            raise RuntimeError("onnxruntime is required for the Autoware CNN backend") from exc
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 2
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), session_options,
            providers=list(providers or ["CPUExecutionProvider"]),
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or tuple(inputs[0].shape[1:]) != (3, *self.input_size[::-1]):
            raise ValueError(f"unexpected Autoware MobileNetV2 input: {[i.shape for i in inputs]}")
        self.input_name = inputs[0].name
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise ValueError(f"unexpected Autoware MobileNetV2 outputs: {[o.shape for o in outputs]}")
        self.output_name = outputs[0].name

    def _prepare(self, roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0 or roi.ndim != 3:
            raise ValueError("empty BGR ROI")
        resized = cv2.resize(roi, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
        return np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1)[None])

    def classify(self, roi: np.ndarray, *, request=None) -> Attributes:
        if roi is None or roi.size == 0 or min(roi.shape[:2]) < 2:
            return Attributes(reason="INSUFFICIENT_PIXELS", backend="autoware_mobilenetv2_v4")
        started = time.perf_counter()
        tensor = self._prepare(roi)
        output = np.asarray(self.session.run([self.output_name], {self.input_name: tensor})[0])
        probabilities, output_semantics = _probabilities(output[0] if output.ndim > 1 else output)
        index = int(np.argmax(probabilities))
        label = self.labels[index] if index < len(self.labels) else "unknown"
        color, shape, directions, reason = parse_label(label)
        confidence = float(probabilities[index])
        top = np.argsort(probabilities)[::-1][: min(3, len(probabilities))]
        evidence = {
            "autoware_label": label,
            "output_semantics": output_semantics,
            "autoware_directions": directions,
            "top_k": [{"label": self.labels[int(i)] if int(i) < len(self.labels) else "unknown",
                       "confidence": float(probabilities[int(i)])} for i in top],
            "native_roi_size": [int(roi.shape[1]), int(roi.shape[0])],
            "analysis_roi_size": list(self.input_size),
            "normalization": {"mean_rgb": [123.675, 116.28, 103.53],
                               "std_rgb": [58.395, 57.12, 57.375]},
            "inference_ms": (time.perf_counter() - started) * 1000.0,
        }
        return Attributes(
            color=color,
            shape=shape,
            color_confidence=confidence if color != "UNKNOWN" else 0.0,
            shape_confidence=confidence if shape != "UNKNOWN" else 0.0,
            reason=reason,
            backend="autoware_mobilenetv2_v4",
            evidence=evidence,
        )


class ConservativeFusionBackend:
    """Use CNN only when it adds evidence without overriding a clear conflict."""

    def __init__(self, traditional, cnn):
        self.traditional = traditional
        self.cnn = cnn

    def classify(self, roi: np.ndarray, *, request=None) -> Attributes:
        legacy = self.traditional.classify(roi, request=request)
        learned = self.cnn.classify(roi, request=request)
        color = legacy.color
        if color == "UNKNOWN" and learned.color != "UNKNOWN":
            color = learned.color
        elif color != "UNKNOWN" and learned.color not in ("UNKNOWN", color):
            color = "UNKNOWN"
        shape = legacy.shape
        if shape == "UNKNOWN" and learned.shape != "UNKNOWN":
            # A color-only CNN label such as ``red`` maps to ROUND in the
            # public interface, but it contains no geometric evidence.  Do
            # not turn a small/blurred ROI into a known round signal.
            if not (learned.shape == "ROUND" and learned.reason == "AUTOWARE_COLOR_ONLY"):
                shape = learned.shape
        elif shape != "UNKNOWN" and learned.shape not in ("UNKNOWN", shape):
            shape = "UNKNOWN"
        return Attributes(
            color=color,
            shape=shape,
            color_confidence=legacy.color_confidence if legacy.color != "UNKNOWN" else learned.color_confidence,
            shape_confidence=legacy.shape_confidence if legacy.shape != "UNKNOWN" else learned.shape_confidence,
            reason="FUSED_AGREE" if color != "UNKNOWN" or shape != "UNKNOWN" else "FUSED_UNKNOWN",
            backend="traditional_plus_autoware_mobilenetv2",
            evidence={"traditional": legacy.evidence, "cnn": learned.evidence,
                      "traditional_reason": legacy.reason, "cnn_reason": learned.reason},
        )
