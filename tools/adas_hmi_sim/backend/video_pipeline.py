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
from .surround_bev import SurroundBEVComposer, load_bev_config


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
    surround_bev: np.ndarray | None = None


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
        self.bev_composer = SurroundBEVComposer()

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
            "surround_bev": self.bev_composer.metadata(),
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
            self.bev_composer.compose(images),
        )

    def close(self) -> None:
        self.primary.close()
        self.bev_composer.close()


class SurroundVideoSource:
    """Replay four synchronized camera files and emit a real BEV image.

    An optional ``detector_model`` runs only on the front stream.  This keeps
    the expensive RKNN stage independent from the four-camera compositor and
    lets the same source run as a PC replay or as an RK3566 smoke test.
    """

    def __init__(self, video_dir: str | Path, bev_config: str | Path | None = None, loop: bool = True, detector_model: str | Path | None = None, detector_fps: float = 10.0, detector_input_size: int | tuple[int, int] = 320, lut_dir: str | Path | None = None):
        root = Path(video_dir)
        candidates = {
            "front": root / "front.mp4",
            "left": root / "left.mp4",
            "right": root / "right.mp4",
            "rear": root / "rear.mp4",
        }
        if not candidates["rear"].exists() and (root / "back.mp4").exists():
            candidates["rear"] = root / "back.mp4"
        missing = [role for role, path in candidates.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing surround videos in {root}: {', '.join(missing)}")
        self.captures = {role: cv2.VideoCapture(str(path)) for role, path in candidates.items()}
        if not all(capture.isOpened() for capture in self.captures.values()):
            raise RuntimeError(f"could not open all surround videos in {root}")
        self.fps = float(self.captures["front"].get(cv2.CAP_PROP_FPS) or 25.0)
        self.loop = loop
        self.frame_id = 0
        if lut_dir is not None:
            from .calibrated_surround import CalibratedSurroundComposer
            self.composer = CalibratedSurroundComposer(lut_dir)
        else:
            self.composer = SurroundBEVComposer(load_bev_config(Path(bev_config) if bev_config else None))
        self.detector_fps = max(0.0, float(detector_fps))
        self._last_detector_s = 0.0
        self.detector = None
        if detector_model is not None:
            model_path = Path(detector_model)
            if model_path.suffix.lower() != ".rknn":
                raise ValueError("surround detector currently accepts an RKNN model so board and PC paths stay explicit")
            from competition_ai.detector import CompetitionDetector
            self.detector = CompetitionDetector(model_path, input_size=detector_input_size, conf_threshold=0.25)

    def next(self) -> FramePacket | None:
        frames: dict[str, np.ndarray] = {}
        for role, capture in self.captures.items():
            ok, frame = capture.read()
            if not ok and self.loop:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                return None
            frames[role] = frame
        timestamp_ms = int(self.frame_id * 1000 / max(1.0, self.fps))
        bev = self.composer.compose(frames)
        detections: list[dict[str, Any]] = []
        detector_update = True
        if self.detector is not None:
            now = time.perf_counter()
            due = self._last_detector_s == 0.0 or self.detector_fps <= 0 or now - self._last_detector_s >= 1.0 / self.detector_fps
            if due:
                result = self.detector.infer(frames["front"], frame_id=self.frame_id, timestamp_ns=timestamp_ms * 1_000_000)
                detections = [item.to_dict() for item in result.detections]
                self._last_detector_s = now
            else:
                detector_update = False
        metadata = {
            "system_mode": "SURROUND_BEV",
            "channel_roles": list(frames),
            "surround_bev": self.composer.metadata(),
            "traffic_light_state": "UNKNOWN",
            "rk3566": {
                "detector": "RKNN" if self.detector is not None else "NOT_CONNECTED",
                "detector_model": str(self.detector.model_path) if self.detector is not None else None,
                "detector_input_size": list(self.detector.input_size) if self.detector is not None and isinstance(self.detector.input_size, tuple) else self.detector.input_size if self.detector is not None else None,
                "detector_fps_target": self.detector_fps if self.detector is not None else 0.0,
            },
        }
        packet = FramePacket(self.frame_id, timestamp_ms, frames["front"], detections, metadata, detector_update, frames, None, bev)
        self.frame_id += 1
        return packet

    def close(self) -> None:
        for capture in self.captures.values():
            capture.release()
        self.composer.close()
        if self.detector is not None:
            self.detector.close()


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
    def __init__(self, video: str | Path, model: str | Path, confidence: float = 0.25, device: str = "cpu", fps_limit: float = 10.0, input_size: int | tuple[int, int] = 640, traffic_light_min_confidence: float = 0.25, traffic_light_hold_frames: int = 4, traffic_light_association: str = "unverified", lane_stride: int = 1, signal_roi: bool = False, signal_roi_confidence: float = 0.10, signal_roi_input_size: int | tuple[int, int] | None = None, signal_roi_top: float = 0.55, traffic_light_model: str | Path | None = None, traffic_light_confidence: float = 0.05, traffic_light_fps: float = 3.0, traffic_light_roi_top: float = 0.60, traffic_light_specialist_mode: str = "fuse", start_frame: int = 0, loop: bool = False, traffic_light_roi_mode: str = "top60"):
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open live video: {video}")
        self.start_frame = max(0, int(start_frame))
        self.loop = bool(loop)
        if self.start_frame > 0:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.index = 0
        self.fps_limit = max(0.0, float(fps_limit))
        self._last_emit = 0.0
        self._last_signal_emit = 0.0
        self._last_signal_detections: list[dict[str, Any]] = []
        self._last_signal_inference_ms: float | None = None
        self._signal_fps = max(0.0, float(traffic_light_fps))
        if traffic_light_specialist_mode not in {"fuse", "replace"}:
            raise ValueError("traffic_light_specialist_mode must be 'fuse' or 'replace'")
        self.traffic_light_specialist_mode = traffic_light_specialist_mode
        self.traffic_light_association = traffic_light_association
        self.signal_roi_confidence = max(0.0, float(signal_roi_confidence))
        self.signal_roi_input_size = signal_roi_input_size if signal_roi_input_size is not None else input_size
        self.signal_roi_top = float(np.clip(signal_roi_top, 0.30, 0.90))
        self._last_result_metadata: dict[str, Any] = {
            "system_mode": "PC_FRONT_LIVE",
            "traffic_light_state": "UNKNOWN",
            "traffic_light_decision_state": "UNKNOWN",
            "traffic_light": {"state": "UNKNOWN", "confidence": 0.0, "bbox": None, "candidate_count": 0, "signal_type": "UNKNOWN", "pedestrian_state": "UNKNOWN"},
            "lanes": {},
            "lane_detector": {"coverage": 0.0, "source": "OPENCV_HOUGH_EXPERIMENTAL", "validation": "EXPERIMENTAL"},
        }
        model_path = Path(model)
        self.signal_roi_enabled = bool(signal_roi and model_path.suffix.lower() == ".pt" and traffic_light_model is None)
        self.signal_detector = None
        self.signal_specialist = None
        from .traffic_light_specialist import SignalCandidateCache
        self.signal_cache = SignalCandidateCache(confidence=traffic_light_confidence)
        self.traffic_light_model_path = Path(traffic_light_model) if traffic_light_model is not None else None
        if traffic_light_model is not None:
            from .traffic_light_specialist import TrafficLightSpecialist

            self.signal_specialist = TrafficLightSpecialist(
                traffic_light_model,
                confidence=traffic_light_confidence,
                roi_top=traffic_light_roi_top,
                device=device,
                roi_mode=traffic_light_roi_mode,
            )
        if model_path.suffix.lower() == ".rknn":
            from competition_ai.detector import CompetitionDetector
            self.detector = CompetitionDetector(model_path, input_size=input_size, conf_threshold=confidence)
        else:
            from adas_edge.perception.ultralytics_detector import UltralyticsDetector
            self.detector = UltralyticsDetector(model_path, confidence=confidence, input_size=input_size, device=device, exact_input_canvas=True)
            if self.signal_roi_enabled:
                # The main detector sees the complete frame. A second PC-only
                # pass enlarges the upper signal corridor without changing
                # the board's fixed 512x288 NPU path.
                self.signal_detector = UltralyticsDetector(model_path, confidence=self.signal_roi_confidence, input_size=self.signal_roi_input_size, device=device)
        from .traffic_light_runtime import TrafficLightTemporalFilter
        from .lane_runtime import FrontLaneCandidate
        self.traffic_filter = TrafficLightTemporalFilter(min_confidence=traffic_light_min_confidence, hold_frames=traffic_light_hold_frames, association_mode=traffic_light_association,
                                                       specialist_min_confidence=traffic_light_confidence, ego_x_max=.95, pedestrian_hold_frames=0)
        self.lane_estimator = FrontLaneCandidate(update_stride=lane_stride)
        self.model_path = model_path
        self.input_size = input_size

    def next(self) -> FramePacket | None:
        ok, image = self.capture.read()
        if not ok and self.loop:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.index = self.start_frame
            if hasattr(self, "traffic_filter"):
                self.traffic_filter.reset()
            if hasattr(self, "lane_estimator"):
                self.lane_estimator.reset()
            self.signal_cache.reset()
            self._last_signal_emit = 0.0
            self._last_signal_detections = []
            if self.signal_specialist is not None:
                self.signal_specialist.reset_roi()
            self._last_result_metadata["traffic_light_state"] = "UNKNOWN"
            self._last_result_metadata["traffic_light_decision_state"] = "UNKNOWN"
            ok, image = self.capture.read()
        if not ok:
            return None
        frame_id = self.index
        self.index += 1
        now = time.perf_counter()
        if self.fps_limit > 0 and self._last_emit and now - self._last_emit < 1.0 / self.fps_limit:
            timestamp_ms = int(frame_id * 1000 / max(1.0, self.capture.get(cv2.CAP_PROP_FPS)))
            return FramePacket(frame_id, timestamp_ms, image, [], dict(self._last_result_metadata), False)
        self._last_emit = now
        timestamp_ms = int(frame_id * 1000 / max(1.0, self.capture.get(cv2.CAP_PROP_FPS)))
        detector_inference_ms = None
        signal_roi_ms = None
        if hasattr(self.detector, "infer"):
            result = self.detector.infer(image, frame_id=frame_id, timestamp_ns=timestamp_ms * 1_000_000)
            detections = [item.to_dict() for item in result.detections]
            detector_inference_ms = round(float(getattr(result, "inference_ms", 0.0)), 3)
        else:
            from adas_edge.contracts import Frame, PixelFormat
            from datetime import datetime, timezone
            frame = Frame("front", frame_id, datetime.now(timezone.utc), image, PixelFormat.BGR888)
            detector_started = time.perf_counter()
            detections = [item.to_dict() for item in self.detector.detect(frame)]
            detector_inference_ms = round((time.perf_counter() - detector_started) * 1000.0, 3)
            if self.signal_detector is not None:
                roi_started = time.perf_counter()
                height, width = image.shape[:2]
                x0, x1 = int(width * 0.10), int(width * 0.90)
                y0, y1 = 0, int(height * self.signal_roi_top)
                crop = image[y0:y1, x0:x1]
                roi_frame = Frame("front_signal_roi", frame_id, datetime.now(timezone.utc), crop, PixelFormat.BGR888)
                roi_items = [item.to_dict() for item in self.signal_detector.detect(roi_frame)]
                for item in roi_items:
                    bbox = item.get("bbox") or item.get("bbox_xyxy")
                    if bbox is None or len(bbox) != 4:
                        continue
                    rx1, ry1, rx2, ry2 = (float(value) for value in bbox)
                    item["bbox"] = [
                        x0 + rx1 * (x1 - x0) / max(1, crop.shape[1]),
                        y0 + ry1 * (y1 - y0) / max(1, crop.shape[0]),
                        x0 + rx2 * (x1 - x0) / max(1, crop.shape[1]),
                        y0 + ry2 * (y1 - y0) / max(1, crop.shape[0]),
                    ]
                detections.extend(item for item in roi_items if item.get("class_name") == "traffic light")
                signal_roi_ms = round((time.perf_counter() - roi_started) * 1000.0, 3)
        if self.signal_specialist is not None:
            # The specialist is intentionally slower than the broad detector.
            # Keep its last boxes between updates so the temporal filter sees a
            # coherent signal track without running a second NPU/PC pass at the
            # main detector rate.
            from .traffic_light_specialist import signal_refresh_due
            signal_due = signal_refresh_due(now, None if self._last_signal_emit == 0. else self._last_signal_emit, self._signal_fps)
            if self.traffic_light_specialist_mode == "replace":
                detections = [
                    item
                    for item in detections
                    if str(item.get("class_name", "")).replace("_", " ").lower() != "traffic light"
                ]
            if signal_due:
                self._last_signal_detections, self._last_signal_inference_ms = self.signal_specialist.detect(image, frame_id=frame_id)
                self.signal_cache.store(image, self._last_signal_detections,
                                        frame_id=frame_id, timestamp_ms=timestamp_ms)
                self._last_signal_emit = now
            detections.extend(self.signal_cache.current(timestamp_ms))
        lane_estimate = self.lane_estimator.update(image)
        estimate = self.traffic_filter.update(image, detections, lane_metadata=lane_estimate.as_metadata())
        self._last_result_metadata = {
            "system_mode": "PC_FRONT_LIVE",
            # SceneFrame displays the measured candidate color, while the
            # decision field remains UNKNOWN until signal-to-ego-lane
            # association is validated.
            "traffic_light_state": estimate.state,
            "traffic_light_decision_state": estimate.decision_state,
            "traffic_light": estimate.as_metadata(),
            "lanes": lane_estimate.lanes,
            "lane_detector": lane_estimate.as_metadata(),
            "perception": {
                "model": self.model_path.name,
                "input_size": list(self.input_size) if isinstance(self.input_size, tuple) else self.input_size,
                "inference_ms": detector_inference_ms,
                "signal_roi_enabled": self.signal_roi_enabled,
                "signal_roi_confidence": self.signal_roi_confidence if self.signal_roi_enabled else None,
                "signal_roi_input_size": list(self.signal_roi_input_size) if isinstance(self.signal_roi_input_size, tuple) else self.signal_roi_input_size if self.signal_roi_enabled else None,
                "signal_roi_top": self.signal_roi_top if self.signal_roi_enabled else None,
                "signal_roi_ms": signal_roi_ms if "signal_roi_ms" in locals() else None,
                "traffic_light_specialist_enabled": self.signal_specialist is not None,
                "traffic_light_specialist_model": self.traffic_light_model_path.name if self.traffic_light_model_path is not None else None,
                "traffic_light_specialist_roi": self.signal_specialist.last_roi if self.signal_specialist is not None else None,
                "traffic_light_specialist_fps_target": self._signal_fps if self.signal_specialist is not None else 0.0,
                "traffic_light_specialist_mode": self.traffic_light_specialist_mode if self.signal_specialist is not None else None,
                "traffic_light_specialist_inference_ms": self._last_signal_inference_ms,
                "traffic_light_specialist_updated": bool(self.signal_specialist is not None and signal_due),
                "traffic_light_association": self.traffic_light_association,
                "detector_frame_id": frame_id,
            },
        }
        packet = FramePacket(frame_id, timestamp_ms, image, detections, dict(self._last_result_metadata))
        return packet

    def close(self) -> None:
        self.capture.release()
        if hasattr(self.detector, "close"):
            self.detector.close()
        if self.signal_detector is not None and hasattr(self.signal_detector, "close"):
            self.signal_detector.close()
        if self.signal_specialist is not None:
            self.signal_specialist.close()
