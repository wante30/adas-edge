#!/usr/bin/env python3
"""Latest-only RKNN/OCR sidecar for the Mali RGB565 replay renderer."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import threading
import time

try:
    import resource
except ImportError:
    resource = None

import cv2
import numpy as np


ROOT = Path(os.environ.get("ADAS_EDGE_ROOT", Path(__file__).resolve().parents[3]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_ai.scenario import Lane
from tools.competition_console.fpga_perception import TrafficCode, build_hud_perception
from tools.competition_console.navigation_reader import NavigationReader
from tools.competition_console.pipeline import FrontPipeline
from tools.competition_console.rknn_detector import RknnConsoleDetector
from tools.competition_console.latest_frame_slot import LatestFrameSlot


WIDTH = 1280
HEIGHT = 720
FRAME_BYTES = WIDTH * HEIGHT * 2
REQUEST_HEADER = struct.Struct("<4sQQ")
RESULT_HEADER = struct.Struct("<4sQQIIB3xf")
BOX = struct.Struct("<7f")
MAX_BOXES = 32
FLAG_RED = 1 << 0
FLAG_GREEN = 1 << 1
FLAG_PEDESTRIAN_HAZARD = 1 << 2
FLAG_OBSTACLE_HAZARD = 1 << 3
FLAG_NAVIGATION_VALID = 1 << 4
NAV_DIRECTIONS = {"none": 0, "left": 1, "right": 2, "forward": 3, "uturn": 4}


def encode_frame_request(frame_id: int, timestamp_ns: int) -> bytes:
    return REQUEST_HEADER.pack(b"MVR1", int(frame_id), int(timestamp_ns))


def decode_frame_request(packet: bytes):
    if len(packet) != REQUEST_HEADER.size:
        return None
    magic, frame_id, timestamp_ns = REQUEST_HEADER.unpack(packet)
    if magic != b"MVR1":
        return None
    return frame_id, timestamp_ns


def encode_ai_result_packet(*, frame_id, timestamp_ns, flags, nav_direction,
                            nav_distance_m, boxes):
    boxes = list(boxes[:MAX_BOXES])
    payload = bytearray(RESULT_HEADER.pack(
        b"MVA1", int(frame_id), int(timestamp_ns), len(boxes), int(flags),
        int(nav_direction), float(nav_distance_m),
    ))
    for box in boxes:
        if len(box) != 7 or not all(math.isfinite(float(value)) for value in box):
            raise ValueError("AI boxes must contain seven finite float values")
        payload.extend(BOX.pack(*(float(value) for value in box)))
    return bytes(payload)


def decode_ai_result_packet(packet: bytes):
    if len(packet) < RESULT_HEADER.size:
        return None
    magic, frame_id, timestamp_ns, count, flags, nav_direction, nav_distance_m = (
        RESULT_HEADER.unpack_from(packet)
    )
    if (magic != b"MVA1" or count > MAX_BOXES
            or len(packet) != RESULT_HEADER.size + count * BOX.size):
        return None
    boxes = [
        BOX.unpack_from(packet, RESULT_HEADER.size + index * BOX.size)
        for index in range(count)
    ]
    if not math.isfinite(nav_distance_m):
        return None
    return {
        "frame_id": frame_id,
        "timestamp_ns": timestamp_ns,
        "flags": flags,
        "nav_direction": nav_direction,
        "nav_distance_m": nav_distance_m,
        "boxes": boxes,
    }


def decode_rgb565_payload(payload: bytes):
    if len(payload) != FRAME_BYTES:
        raise ValueError(f"RGB565 frame must be {FRAME_BYTES} bytes, got {len(payload)}")
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)
    return cv2.cvtColor(packed, cv2.COLOR_BGR5652BGR)


def load_replay_frames(paths):
    started = time.perf_counter()
    frames = []
    for path in paths:
        payload = Path(path).read_bytes()
        frames.append(decode_rgb565_payload(payload))
    return frames, (time.perf_counter() - started) * 1000


def crop_quadrant(image, quadrant):
    if quadrant not in (0, 1, 2, 3):
        raise ValueError("quadrant must be 0, 1, 2, or 3")
    height, width = image.shape[:2]
    row, col = divmod(quadrant, 2)
    return image[row * height // 2:(row + 1) * height // 2,
                 col * width // 2:(col + 1) * width // 2]


def _distribution(values, unit="ms"):
    if not values:
        return {"samples": 0, "p50_" + unit: None, "p95_" + unit: None,
                "mean_" + unit: None}
    samples = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(len(samples)),
        "p50_" + unit: float(np.percentile(samples, 50)),
        "p95_" + unit: float(np.percentile(samples, 95)),
        "mean_" + unit: float(np.mean(samples)),
    }


class BoardTelemetry:
    def __init__(self):
        self.samples = {"npu_load_percent": [], "npu_frequency_mhz": [],
                        "gpu_load_percent": [], "gpu_frequency_mhz": []}
        self.temperatures = []
        self.running = threading.Event()
        self.worker = None
        self.zones = []
        for index in range(16):
            base = Path(f"/sys/class/thermal/thermal_zone{index}")
            try:
                name = (base / "type").read_text(encoding="ascii").strip()
                (base / "temp").read_text(encoding="ascii")
            except OSError:
                continue
            self.zones.append((name, base / "temp"))

    @staticmethod
    def _number(path):
        try:
            return float(Path(path).read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _load(path):
        try:
            value = Path(path).read_text(encoding="ascii").strip().split("@", 1)[0]
            return float(value)
        except (OSError, ValueError):
            return None

    def start(self):
        self.running.set()
        self.worker = threading.Thread(target=self._sample, name="board-telemetry", daemon=True)
        self.worker.start()

    def _sample(self):
        roots = {
            "npu": "/sys/class/devfreq/fde40000.npu",
            "gpu": "/sys/class/devfreq/fde60000.gpu",
        }
        while self.running.is_set():
            for device, root in roots.items():
                load = self._load(f"{root}/load")
                frequency = self._number(f"{root}/cur_freq")
                if load is not None:
                    self.samples[f"{device}_load_percent"].append(load)
                if frequency is not None:
                    self.samples[f"{device}_frequency_mhz"].append(frequency / 1e6)
            row = {name: [] for name, _ in self.zones}
            for name, path in self.zones:
                value = self._number(path)
                if value is not None:
                    row[name].append(value / 1000.0)
            self.temperatures.append(row)
            time.sleep(0.25)

    def stop(self):
        self.running.clear()
        if self.worker is not None:
            self.worker.join(timeout=1.0)

    def summary(self):
        devices = {
            device: {
                "load_percent": _distribution(
                    self.samples[f"{device}_load_percent"], "percent"
                ),
                "frequency_mhz": _distribution(
                    self.samples[f"{device}_frequency_mhz"], "mhz"
                ),
            }
            for device in ("npu", "gpu")
        }
        zones = {}
        for name, _ in self.zones:
            values = [row[name][0] for row in self.temperatures if row.get(name)]
            if values:
                zones[name] = {
                    "samples": len(values), "min_c": min(values),
                    "max_c": max(values), "end_c": values[-1],
                }
        return {"devices": devices, "temperatures_c": zones,
                "sample_period_ms": 250}


def _current_rss_mb():
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _quadrant_offset(quadrant):
    row, col = divmod(quadrant, 2)
    return col * (WIDTH // 2), row * (HEIGHT // 2)


def _box_color(name):
    if name == "traffic light":
        return 1.0, 0.68, 0.12
    if name == "person":
        return 1.0, 0.18, 0.18
    return 0.18, 0.92, 0.48


class AiReplayBridge:
    def __init__(self, args):
        self.args = args
        self.frames, self.source_prepare_ms = load_replay_frames(args.files)
        if not self.frames:
            raise ValueError("At least one replay frame is required")

        main = RknnConsoleDetector(args.main_model, (512, 288))
        try:
            specialist = RknnConsoleDetector(
                args.specialist_model, (512, 192), specialist=True
            )
        except BaseException:
            main.close()
            raise
        self.pipeline = FrontPipeline(
            main, specialist, main_hz=args.main_hz,
            specialist_hz=args.specialist_hz, hazards=True,
        )
        self.navigation = NavigationReader()
        self.navigation_last_update_ns = None
        self.navigation_timings = []
        self.latest = LatestFrameSlot()
        self.running = threading.Event()
        self.running.set()
        self.request_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.request_socket.bind(("127.0.0.1", args.request_port))
        self.request_socket.settimeout(0.2)
        self.result_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.result_address = ("127.0.0.1", args.result_port)
        self.listener = threading.Thread(target=self._listen, name="mali-ai-requests", daemon=True)
        self.worker = threading.Thread(target=self._infer, name="mali-ai-worker", daemon=True)
        self.request_count = 0
        self.replaced_count = 0
        self.result_count = 0
        self.error_count = 0
        self.frame_ids = []
        self.request_times = []
        self.latency_ms = []
        self.worker_cpu_ns = 0
        self.process_cpu_start_ns = time.process_time_ns()
        self.telemetry = BoardTelemetry()
        self.first_request_ns = None
        self.last_request_ns = None
        self.last_error = None

    def start(self):
        self.listener.start()
        self.worker.start()

    def _listen(self):
        while self.running.is_set():
            try:
                packet, _ = self.request_socket.recvfrom(128)
            except socket.timeout:
                continue
            except OSError:
                break
            request = decode_frame_request(packet)
            if request is None:
                continue
            now = time.monotonic_ns()
            self.request_count += 1
            self.request_times.append(now)
            if self.first_request_ns is None:
                self.first_request_ns = now
            self.last_request_ns = now
            if self.latest.publish(request):
                self.replaced_count += 1

    def _detections_to_boxes(self, snapshot):
        offset_x, offset_y = _quadrant_offset(self.args.road_quadrant)
        boxes = []
        for detections in (snapshot.main, snapshot.specialist):
            for item in detections or ():
                if len(boxes) >= MAX_BOXES:
                    return boxes
                raw = item.get("bbox")
                if raw is None or len(raw) != 4:
                    continue
                x1, y1, x2, y2 = (float(value) for value in raw)
                if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                    continue
                r, g, b = _box_color(item.get("class_name", ""))
                boxes.append((x1 + offset_x, y1 + offset_y,
                              x2 + offset_x, y2 + offset_y, r, g, b))
        return boxes

    def _analyze(self, frame_id, timestamp_ns):
        full_frame = self.frames[frame_id % len(self.frames)]
        road = crop_quadrant(full_frame, self.args.road_quadrant)
        navigation = crop_quadrant(full_frame, self.args.navigation_quadrant)
        self.pipeline.advance(road, int(frame_id), int(timestamp_ns))
        if (self.navigation_last_update_ns is None or
                timestamp_ns - self.navigation_last_update_ns >= round(1e9 / self.args.ocr_hz)):
            navigation = cv2.resize(navigation, (960, 540), interpolation=cv2.INTER_LINEAR)
            nav_result = self.navigation.update(navigation, int(frame_id), int(timestamp_ns))
            self.navigation_timings.append(float(nav_result.inference_ms))
            self.navigation_last_update_ns = int(timestamp_ns)
        lane = next(iter(Lane))
        snapshot = self.pipeline.snapshot(lane, int(timestamp_ns))
        nav = self.navigation.current(int(timestamp_ns)).to_dict()
        hud = build_hud_perception(road, snapshot)
        flags = 0
        if hud.traffic == TrafficCode.RED:
            flags |= FLAG_RED
        elif hud.traffic == TrafficCode.GREEN:
            flags |= FLAG_GREEN
        if snapshot.hazards.pedestrian_hazard:
            flags |= FLAG_PEDESTRIAN_HAZARD
        if snapshot.hazards.obstacle_hazard:
            flags |= FLAG_OBSTACLE_HAZARD
        if nav.get("valid"):
            flags |= FLAG_NAVIGATION_VALID
        nav_direction = NAV_DIRECTIONS.get(nav.get("direction", "none"), 0)
        nav_distance = nav.get("distance_m")
        boxes = self._detections_to_boxes(snapshot)
        return encode_ai_result_packet(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            flags=flags,
            nav_direction=nav_direction,
            nav_distance_m=float(nav_distance) if nav_distance is not None else -1.0,
            boxes=boxes,
        )

    def _infer(self):
        while self.running.is_set():
            request = self.latest.take(timeout=0.2)
            if request is None:
                continue
            frame_id, timestamp_ns = request
            started = time.perf_counter_ns()
            cpu_started = time.thread_time_ns()
            try:
                packet = self._analyze(frame_id, timestamp_ns)
                self.result_socket.sendto(packet, self.result_address)
                self.result_count += 1
                self.frame_ids.append(frame_id)
                self.latency_ms.append((time.perf_counter_ns() - started) / 1e6)
            except BaseException as exc:
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.worker_cpu_ns += time.thread_time_ns() - cpu_started

    def run(self):
        self.telemetry.start()
        self.start()
        print(
            f"READY evidence=LOCAL_REPLAY request_port={self.args.request_port} "
            f"result_port={self.args.result_port} frames={len(self.frames)}",
            flush=True,
        )
        try:
            while self.running.is_set():
                time.sleep(0.1)
                if (self.last_request_ns is not None and
                        time.monotonic_ns() - self.last_request_ns >
                        int(self.args.idle_timeout * 1e9)):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        self.running.clear()
        self.latest.close()
        try:
            self.request_socket.close()
        except OSError:
            pass
        self.listener.join(timeout=1.0)
        self.worker.join(timeout=10.0)
        self.result_socket.close()
        self.telemetry.stop()
        self.pipeline.main_detector.close()
        self.pipeline.specialist_detector.close()
        self._write_summary()

    def _write_summary(self):
        duration_ns = 0
        if self.first_request_ns is not None and self.last_request_ns is not None:
            duration_ns = self.last_request_ns - self.first_request_ns
        timings = {
            "main_rknn": _distribution(self.pipeline.main_timings),
            "specialist_rknn": _distribution(self.pipeline.specialist_timings),
            "navigation_ocr": _distribution(self.navigation_timings),
            "hazard_postprocess": _distribution(
                self.pipeline.hazard_engine.timings
                if self.pipeline.hazard_engine is not None else []
            ),
            "all_ai_work": _distribution(self.latency_ms),
        }
        peak_rss_mb = None
        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            peak_rss_mb = float(usage.ru_maxrss) / 1024.0
        duration_s = duration_ns / 1e9 if duration_ns else 0.0
        process_cpu_ns = time.process_time_ns() - self.process_cpu_start_ns
        online_cpus = os.cpu_count() or 1
        rknpu_updates = len(self.pipeline.main_timings)
        specialist_updates = len(self.pipeline.specialist_timings)
        ocr_updates = len(self.navigation_timings)
        report = {
            "evidence": "LOCAL_REPLAY",
            "frame_format": "1280x720 RGB565_LE packed, decoded using the existing PGV BGR565 conversion",
            "model_paths": {
                "yolo11n": str(self.args.main_model),
                "traffic_light_specialist": str(self.args.specialist_model),
            },
            "source_frames_loaded": len(self.frames),
            "source_prepare_ms": self.source_prepare_ms,
            "input_requests_received": self.request_count,
            "input_request_fps": self.request_count / duration_s if duration_s else 0.0,
            "source_ids_analyzed": len(self.frame_ids),
            "source_id_analysis_fps": len(self.frame_ids) / duration_s if duration_s else 0.0,
            "source_ids_replaced_before_analysis": self.replaced_count,
            "result_packets_sent": self.result_count,
            "inference_errors": self.error_count,
            "last_error": self.last_error,
            "updates_per_second": {
                "yolo11n_rknn": rknpu_updates / duration_s if duration_s else 0.0,
                "traffic_light_specialist_rknn": specialist_updates / duration_s if duration_s else 0.0,
                "navigation_ocr": ocr_updates / duration_s if duration_s else 0.0,
            },
            "timings_ms": timings,
            "resources": {
                "process_cpu_one_core_percent": 100.0 * process_cpu_ns / duration_ns if duration_ns else 0.0,
                "process_cpu_machine_percent": (
                    100.0 * process_cpu_ns / duration_ns / online_cpus if duration_ns else 0.0
                ),
                "inference_worker_cpu_one_core_percent": (
                    100.0 * self.worker_cpu_ns / duration_ns if duration_ns else 0.0
                ),
                "online_cpus": online_cpus,
                "rss_end_mb": _current_rss_mb(),
                "rss_peak_mb": peak_rss_mb,
                "board_telemetry": self.telemetry.summary(),
            },
            "display_note": "AI runs in a separate latest-only worker and never blocks EGL frame submission.",
        }
        path = self.args.summary
        if path:
            Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


def _read_npu_load_samples():
    samples = []
    for path in ("/sys/kernel/debug/rknpu/load", "/sys/class/devfreq/fde40000.npu/load"):
        try:
            value = Path(path).read_text(encoding="ascii").strip()
        except OSError:
            continue
        samples.append({"path": path, "value": value})
    return samples


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--file", dest="files", action="append", required=True)
    result.add_argument("--main-model", required=True)
    result.add_argument("--specialist-model", required=True)
    result.add_argument("--request-port", type=int, default=38260)
    result.add_argument("--result-port", type=int, default=38261)
    result.add_argument("--road-quadrant", type=int, choices=range(4), default=3)
    result.add_argument("--navigation-quadrant", type=int, choices=range(4), default=1)
    result.add_argument("--main-hz", type=float, default=10.0)
    result.add_argument("--specialist-hz", type=float, default=3.0)
    result.add_argument("--ocr-hz", type=float, default=3.0)
    result.add_argument("--idle-timeout", type=float, default=2.0)
    result.add_argument("--summary", default="")
    return result


def main():
    args = parser().parse_args()
    bridge = AiReplayBridge(args)
    bridge.run()


if __name__ == "__main__":
    main()
