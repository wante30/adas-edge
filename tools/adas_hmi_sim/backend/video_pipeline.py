"""Input adapters for mock, replay, and live PC perception."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from competition_io import FrameDescriptor, MockRoundRobin, MockTransport, ResultDescriptor


@dataclass
class FramePacket:
    frame_id: int
    timestamp_ms: int
    image: np.ndarray | None
    detections: list[dict[str, Any]]
    metadata: dict[str, Any]
    detector_update: bool = True
    stream_images: dict[str, np.ndarray] | None = None
    transport: dict[str, Any] | None = None


def _detection(class_name: str, confidence: float, bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    return {"class_name": class_name, "confidence": confidence, "bbox": list(bbox)}


def _road_frame(width: int, height: int, detections: list[dict[str, Any]], traffic: str, timestamp_ms: int) -> np.ndarray:
    """Draw a compact synthetic front camera view so mock mode is self-contained."""

    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        value = int(20 + 32 * y / height)
        image[y, :, :] = (value // 2, value, min(75, value + 10))
    horizon = int(height * 0.48)
    cv2.rectangle(image, (0, horizon), (width, height), (35, 39, 43), -1)
    cv2.fillPoly(image, [np.array([(width // 2 - 42, horizon), (width // 2 + 42, horizon), (width + 130, height), (-130, height)], dtype=np.int32)], (45, 48, 51))
    for offset in (-1, 1):
        cv2.line(image, (width // 2 + offset * 42, horizon), (width // 2 + offset * 230, height), (205, 208, 199), 3)
    cv2.line(image, (width // 2 - 42, horizon), (width // 2 - 230, height), (210, 214, 204), 4)
    cv2.line(image, (width // 2 + 42, horizon), (width // 2 + 230, height), (210, 214, 204), 4)
    # Synthetic traffic signal.
    if traffic != "UNKNOWN":
        center = (width // 2, int(height * 0.18))
        cv2.rectangle(image, (center[0] - 16, center[1] - 42), (center[0] + 16, center[1] + 42), (15, 17, 19), -1)
        colors = {"RED": (35, 45, 235), "YELLOW": (30, 208, 240), "GREEN": (50, 210, 100)}
        cv2.circle(image, center, 11, colors.get(traffic, (90, 90, 90)), -1)
    cv2.putText(image, "FRONT SENSOR / PC SIM", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175, 191, 187), 1, cv2.LINE_AA)
    cv2.putText(image, f"{timestamp_ms / 1000:05.1f}s", (width - 92, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (175, 191, 187), 1, cv2.LINE_AA)
    for item in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in item["bbox"]]
        color = (74, 206, 233) if item["class_name"] != "person" else (100, 180, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{item['class_name']} {item['confidence']:.2f}", (x1, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return image


class MockSource:
    def __init__(self, fps: float = 20.0, duration_s: float = 20.0, width: int = 960, height: int = 540, start_ms: int = 0):
        self.fps, self.duration_s, self.width, self.height = fps, duration_s, width, height
        self.total_frames = max(1, int(fps * duration_s))
        self.index = int(max(0, start_ms) * fps / 1000) % self.total_frames

    def next(self) -> FramePacket | None:
        if self.index >= self.total_frames:
            self.index = 0
        frame_id = self.index
        timestamp_ms = int(frame_id * 1000 / self.fps)
        t = timestamp_ms / 1000
        detections: list[dict[str, Any]] = []
        # Two steady lead cars, with a small apparent approach near the red-light section.
        if t < 20:
            lead_y = 258 + int(max(0.0, min(1.0, (t - 5) / 5)) * 28) if t >= 4 else 248
            detections.extend([
                _detection("car", 0.94, (350, lead_y, 435, min(470, lead_y + 120))),
                _detection("car", 0.88, (540, lead_y + 12, 625, min(490, lead_y + 132))),
            ])
        traffic = "RED" if 4 <= t < 8 else "GREEN" if 8 <= t < 20 else "UNKNOWN"
        if 11 <= t < 15:
            detections.append(_detection("person", 0.91, (448, 278, 500, 472)))
        if 15 <= t:
            detections.append(_detection("car", 0.86, (235, 275, 325, 410)))
        lane_meta = {"left": {"style": "dashed" if t >= 15 else "solid", "confidence": 0.88}, "right": {"style": "dashed", "confidence": 0.87}}
        metadata = {"traffic_light_state": traffic, "lanes": lane_meta, "navigation": {"start_ms": timestamp_ms}}
        image = _road_frame(self.width, self.height, detections, traffic, timestamp_ms)
        self.index += 1
        return FramePacket(frame_id, timestamp_ms, image, detections, metadata)

    def close(self) -> None:
        pass


def _channel_variant(image: np.ndarray, role: str, timestamp_ms: int) -> np.ndarray:
    """Create a deterministic camera preview for the PC four-stream simulator."""

    if role == "rear":
        result = cv2.flip(image, -1)
    elif role == "left":
        result = cv2.convertScaleAbs(image, alpha=0.86, beta=8)
        result = np.roll(result, -36, axis=1)
    elif role == "right":
        result = cv2.convertScaleAbs(image, alpha=0.95, beta=-4)
        result = np.roll(result, 36, axis=1)
    else:
        result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (7, 14, 18), -1)
    cv2.putText(result, f"{role.upper()} / FPGA PCIE CH", (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175, 191, 187), 1, cv2.LINE_AA)
    cv2.putText(result, f"{timestamp_ms / 1000:05.1f}s", (result.shape[1] - 92, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (175, 191, 187), 1, cv2.LINE_AA)
    return result


def _jpeg_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        raise RuntimeError("could not encode a PCIE preview frame")
    return encoded.tobytes()


class FourStreamMockSource:
    """PC-only FPGA→PCIe→RK3566 simulation with four synchronized cameras.

    The images are deterministic mock frames, but every channel crosses the
    existing ``FrameDescriptor``/``ResultDescriptor`` wire contract through an
    in-memory transport. This makes the browser demo exercise the same stage
    boundaries without claiming that a PCIe endpoint or RK3566 is connected.
    """

    ROLES = ("front", "left", "right", "rear")

    def __init__(self, fps: float = 20.0, duration_s: float = 20.0, width: int = 960, height: int = 540, start_ms: int = 0, queue_depth: int = 3):
        self.primary = MockSource(fps=fps, duration_s=duration_s, width=width, height=height, start_ms=start_ms)
        self.width, self.height, self.queue_depth = width, height, queue_depth
        self.frame_transport = MockTransport()
        self.result_transport = MockTransport()
        self.round_robin = MockRoundRobin(streams=4, queue_depth=queue_depth)
        self.pcie_bytes = 0
        self.result_bytes = 0
        self.frames = 0

    def next(self) -> FramePacket | None:
        primary = self.primary.next()
        if primary is None or primary.image is None:
            return None
        images = {role: _channel_variant(primary.image, role, primary.timestamp_ms) for role in self.ROLES}
        descriptor_order: list[int] = []
        descriptor_sizes: dict[str, int] = {}
        for stream_id, role in enumerate(self.ROLES):
            payload = _jpeg_bytes(images[role])
            descriptor = FrameDescriptor(stream_id, primary.frame_id, primary.timestamp_ms * 1_000_000, self.width, self.height, 0xF0, 0, payload, flags=1)
            wire = descriptor.serialize()
            self.pcie_bytes += len(wire)
            descriptor_sizes[role] = len(wire)
            self.frame_transport.send_frame(descriptor)
            self.round_robin.push(descriptor)
        while True:
            descriptor = self.round_robin.pop()
            if descriptor is None:
                break
            # Deserialize the queued transport copy to exercise header checks.
            self.frame_transport.recv_frame(descriptor.stream_id)
            descriptor_order.append(descriptor.stream_id)

        # The mock RK stage emits the front detections as metadata-only results.
        class_ids = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
        records = []
        for item in primary.detections:
            if item.get("class_name") not in class_ids:
                continue
            x1, y1, x2, y2 = [max(0, min(65535, int(round(v)))) for v in item["bbox"]]
            from competition_io.protocol import DetectionRecord
            records.append(DetectionRecord(class_ids[item["class_name"]], float(item["confidence"]), x1, y1, x2, y2))
        result = ResultDescriptor(0, primary.frame_id, primary.timestamp_ms * 1_000_000, tuple(records))
        result_wire = result.serialize()
        self.result_bytes += len(result_wire)
        self.result_transport.send_result(result)
        self.result_transport.recv_result(0)
        self.frames += 1
        metadata = dict(primary.metadata)
        metadata.update({
            "system_mode": "PC_FOUR_STREAM",
            "channel_roles": list(self.ROLES),
            "pcie": {
                "transport": "MockTransport",
                "wire_protocol": "ADSF/ADSR v1",
                "queue_depth": self.queue_depth,
                "descriptor_order": descriptor_order,
                "frame_descriptor_bytes": descriptor_sizes,
                "frame_bytes_total": self.pcie_bytes,
                "result_bytes_total": self.result_bytes,
            },
            "rk3566": {"device": "RK3566_SIM", "detector": "MOCK", "stream_count": 4},
        })
        return FramePacket(
            primary.frame_id,
            primary.timestamp_ms,
            images["front"],
            primary.detections,
            metadata,
            True,
            images,
            {"stream_count": 4, "descriptor_order": descriptor_order, "pcie_bytes": self.pcie_bytes, "result_bytes": self.result_bytes},
        )

    def close(self) -> None:
        self.primary.close()


def _parse_detection(item: dict[str, Any]) -> dict[str, Any]:
    bbox = item.get("bbox") or item.get("bbox_xyxy")
    if bbox is None:
        bbox = [item.get(key, 0) for key in ("x1", "y1", "x2", "y2")]
    return {"class_name": item.get("class_name", "unknown"), "confidence": float(item.get("confidence", 0)), "bbox": bbox}


class ReplaySource:
    def __init__(self, video: str | Path, detections_jsonl: str | Path, loop: bool = False):
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open replay video: {video}")
        self.rows = [json.loads(line) for line in Path(detections_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.loop, self.index = loop, 0
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)

    def next(self) -> FramePacket | None:
        ok, image = self.capture.read()
        if not ok:
            if not self.loop:
                return None
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.index = 0
            ok, image = self.capture.read()
            if not ok:
                return None
        row = self.rows[min(self.index, len(self.rows) - 1)] if self.rows else {}
        frame_id = int(row.get("frame_id", self.index))
        timestamp_ms = int(row.get("timestamp_ms", row.get("timestamp_ns", int(frame_id * 1000 / self.fps)) // 1_000_000))
        metadata = dict(row)
        detections = [_parse_detection(item) for item in row.get("detections", [])]
        self.index += 1
        return FramePacket(frame_id, timestamp_ms, image, detections, metadata)

    def close(self) -> None:
        self.capture.release()


class LiveSource:
    def __init__(self, video: str | Path, model: str | Path, confidence: float = 0.25, device: str = "cpu", fps_limit: float = 10.0):
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open live video: {video}")
        self.index = 0
        self.fps_limit = fps_limit
        self._last_emit = 0.0
        model_path = Path(model)
        if model_path.suffix.lower() == ".rknn":
            from competition_ai.detector import CompetitionDetector
            self.detector = CompetitionDetector(model_path, conf_threshold=confidence)
        else:
            from adas_edge.perception.ultralytics_detector import UltralyticsDetector
            self.detector = UltralyticsDetector(model_path, confidence=confidence, device=device)

    def next(self) -> FramePacket | None:
        ok, image = self.capture.read()
        if not ok:
            return None
        now = time.perf_counter()
        if self.fps_limit > 0 and self._last_emit and now - self._last_emit < 1.0 / self.fps_limit:
            return FramePacket(self.index, int(self.index * 1000 / max(1.0, self.capture.get(cv2.CAP_PROP_FPS))), image, [], {}, False)
        self._last_emit = now
        timestamp_ms = int(self.index * 1000 / max(1.0, self.capture.get(cv2.CAP_PROP_FPS)))
        if hasattr(self.detector, "infer"):
            result = self.detector.infer(image, frame_id=self.index, timestamp_ns=timestamp_ms * 1_000_000)
            detections = [item.to_dict() for item in result.detections]
        else:
            from adas_edge.contracts import Frame, PixelFormat
            from datetime import datetime, timezone
            frame = Frame("front", self.index, datetime.now(timezone.utc), image, PixelFormat.BGR888)
            detections = [item.to_dict() for item in self.detector.detect(frame)]
        packet = FramePacket(self.index, timestamp_ms, image, detections, {})
        self.index += 1
        return packet

    def close(self) -> None:
        self.capture.release()
        if hasattr(self.detector, "close"):
            self.detector.close()
