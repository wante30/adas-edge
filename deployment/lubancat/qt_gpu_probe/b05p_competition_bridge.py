"""Run frozen B05P RKNN candidates through the board's competition AI bridge."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

import cv2
import numpy as np


MODEL_FILES = {
    "bdd": "bdd_yolov8n_512x288_raw_heads_int8.rknn",
    "p2": "b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn",
    "specialist": "b05l_v8n_p2_raw_heads_int8_500_20261026.rknn",
}
REPLAY_ROAD_QUADRANT = 3


def _replay_quadrant_args() -> list[str]:
    return ["--road-quadrant", str(REPLAY_ROAD_QUADRANT)]


def _nchw(value: np.ndarray, channels: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError(f"{label} must be [1,C,H,W] or [1,H,W,C], got {array.shape}")
    if array.shape[1] == channels:
        return array
    if array.shape[-1] == channels:
        return array.transpose(0, 3, 1, 2)
    raise ValueError(f"{label} has no channel dimension of size {channels}: {array.shape}")


def decode_raw_heads_thresholded(
    outputs,
    *,
    input_width: int,
    input_height: int,
    confidence: float,
    iou: float,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> list[list[float]]:
    """Decode only anchors that pass the production confidence threshold."""
    if len(outputs) < 3 or len(outputs) % 3:
        raise ValueError(f"expected three outputs per scale, got {len(outputs)}")
    if not 0.0 <= confidence <= 1.0 or not 0.0 < iou < 1.0:
        raise ValueError("invalid confidence or NMS threshold")
    if input_width <= 0 or input_height <= 0 or scale <= 0:
        raise ValueError("invalid model input transform")

    by_class: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    weights = np.arange(16, dtype=np.float32)
    for index in range(0, len(outputs), 3):
        box = _nchw(outputs[index], 64, f"scale {index // 3} regression")
        _, _, height, width = box.shape
        if height <= 0 or width <= 0:
            raise ValueError("empty output grid")

        scores = np.asarray(outputs[index + 1], dtype=np.float32)
        if scores.ndim == 3:
            scores = scores[None]
        if scores.ndim != 4 or scores.shape[0] != 1:
            raise ValueError(f"invalid class output shape: {scores.shape}")
        if scores.shape[2:] == (height, width):
            pass
        elif scores.shape[1:3] == (height, width):
            scores = scores.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"class grid does not match regression grid: {scores.shape}")

        score_sum = np.asarray(outputs[index + 2], dtype=np.float32)
        if score_sum.ndim == 3:
            score_sum = score_sum[None]
        if score_sum.ndim == 4 and score_sum.shape[1:3] == (height, width) and score_sum.shape[-1] == 1:
            score_sum = score_sum.transpose(0, 3, 1, 2)
        if score_sum.shape != (1, 1, height, width):
            raise ValueError(f"invalid score-sum shape: {score_sum.shape}")

        flat_scores = scores[0].reshape(scores.shape[1], -1)
        possible = np.flatnonzero(flat_scores.max(axis=0) >= confidence)
        if not possible.size:
            continue
        possible_scores = flat_scores[:, possible]
        local_class_ids = possible_scores.argmax(axis=0)
        possible_confidences = possible_scores[local_class_ids, np.arange(possible.size)]
        accepted = possible_confidences >= confidence
        anchor_ids = possible[accepted]
        if not anchor_ids.size:
            continue
        class_ids = local_class_ids[accepted]
        confidences = possible_confidences[accepted]

        logits = box[0].reshape(4, 16, -1)[:, :, anchor_ids].transpose(2, 0, 1)
        logits -= logits.max(axis=2, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=2, keepdims=True)
        distances = (probabilities * weights).sum(axis=2)

        grid_x = (anchor_ids % width).astype(np.float32)
        grid_y = (anchor_ids // width).astype(np.float32)
        stride_x, stride_y = input_width / width, input_height / height
        boxes = np.column_stack((
            (grid_x + 0.5 - distances[:, 0]) * stride_x,
            (grid_y + 0.5 - distances[:, 1]) * stride_y,
            (grid_x + 0.5 + distances[:, 2]) * stride_x,
            (grid_y + 0.5 + distances[:, 3]) * stride_y,
        ))
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, class_ids, confidences = boxes[valid], class_ids[valid], confidences[valid]
        for class_id in np.unique(class_ids):
            mask = class_ids == class_id
            by_class.setdefault(int(class_id), []).append((boxes[mask], confidences[mask]))

    result: list[list[float]] = []
    for class_id, rows in by_class.items():
        boxes = np.concatenate([row[0] for row in rows], axis=0)
        confidences = np.concatenate([row[1] for row in rows], axis=0)
        xywh = boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(), confidences.astype(float).tolist(), 0.0, float(iou)
        )
        if indices is None or len(indices) == 0:
            continue
        for selected in np.asarray(indices).reshape(-1):
            x1, y1, x2, y2 = boxes[int(selected)]
            result.append([
                float(x1), float(y1), float(x2), float(y2),
                float(confidences[int(selected)]), float(class_id),
            ])
    return result


def _stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not values.size:
        return {"samples": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "samples": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_replay_frames(data_root: Path, replay_dir: Path, count: int) -> list[Path]:
    if replay_dir.exists():
        frames = sorted(replay_dir.glob("frame_*.rgb565"))
        if len(frames) != count:
            raise FileExistsError(f"replay directory exists with {len(frames)} frames: {replay_dir}")
        return frames
    image_paths = sorted(
        (data_root / "images").glob("*.png"), key=lambda path: path.name.lower()
    )
    if len(image_paths) < count:
        raise FileNotFoundError(f"need {count} Cityscapes images, found {len(image_paths)}")
    indices = np.linspace(0, len(image_paths) - 1, count, dtype=np.int32)
    replay_dir.mkdir(parents=True)
    paths = []
    for output_index, image_index in enumerate(indices):
        image = cv2.imread(str(image_paths[int(image_index)]))
        if image is None:
            raise ValueError(f"cannot read {image_paths[int(image_index)]}")
        road = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        row, column = divmod(REPLAY_ROAD_QUADRANT, 2)
        frame[row * 360:(row + 1) * 360,
              column * 640:(column + 1) * 640] = road
        packed = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
        output = replay_dir / f"frame_{output_index:03d}.rgb565"
        output.write_bytes(packed.tobytes())
        paths.append(output)
    return paths


def _unused_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _install_raw_head_adapter(bridge_module, scheme: str):
    detector_module = __import__("competition_ai.detector", fromlist=["CompetitionDetector"])
    console_module = __import__(
        "tools.competition_console.rknn_detector", fromlist=["COMPETITION_CLASSES"]
    )
    base_detector = detector_module.CompetitionDetector
    competition_classes = console_module.COMPETITION_CLASSES
    detection_type = __import__("competition_ai.types", fromlist=["Detection"]).Detection
    result_type = __import__("competition_ai.types", fromlist=["FrameResult"]).FrameResult

    class RawHeadCompetitionDetector(base_detector):
        def infer(self, frame, stream_id=0, frame_id=0, timestamp_ns=None):
            preprocess_started = time.perf_counter_ns()
            image, scale, pad_x, pad_y = self.preprocess(frame)
            preprocess_ms = (time.perf_counter_ns() - preprocess_started) / 1e6
            rknn_started = time.perf_counter_ns()
            outputs = self._runner.inference(inputs=[image[None, ...]], data_format="nhwc")
            rknn_call_ms = (time.perf_counter_ns() - rknn_started) / 1e6
            post_started = time.perf_counter_ns()
            rows = decode_raw_heads_thresholded(
                outputs,
                input_width=int(self.input_size[0]),
                input_height=int(self.input_size[1]),
                confidence=self.conf_threshold,
                iou=self.nms_threshold,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
            )
            decode_nms_ms = (time.perf_counter_ns() - post_started) / 1e6
            total_ms = preprocess_ms + rknn_call_ms + decode_nms_ms
            self.last_timing = {
                "preprocess_ms": preprocess_ms,
                "rknn_call_ms": rknn_call_ms,
                "decode_nms_ms": decode_nms_ms,
                "total_ms": total_ms,
                "output_count": len(outputs),
                "detection_count": len(rows),
                "preprocess_backend": getattr(self, "last_preprocess_backend", "opencv"),
            }
            history = getattr(self, "timing_history", None)
            if history is None:
                history = deque(maxlen=3600)
                self.timing_history = history
            history.append(self.last_timing)
            detections = [
                detection_type(
                    int(row[5]),
                    self.class_names[int(row[5])] if 0 <= int(row[5]) < len(self.class_names) else f"class_{int(row[5])}",
                    float(row[4]), *map(float, row[:4]),
                )
                for row in rows
            ]
            return result_type(stream_id, int(frame_id), timestamp_ns,
                               rknn_call_ms + decode_nms_ms, detections)

    detector_module.CompetitionDetector = RawHeadCompetitionDetector
    base_console_detector = bridge_module.RknnConsoleDetector
    single_model = scheme in {"A", "B"}
    p2_model = scheme in {"B", "D"}

    class CompetitionConsoleDetector(base_console_detector):
        def __init__(self, model_path, input_size, *, specialist=False,
                     class_names=None, class_profile=None):
            self.single_model = single_model
            self.specialist = specialist
            self.source = "TL_SPECIALIST_RKNN" if specialist else "MAIN_RKNN"
            self.frame_id = 0
            if p2_model and not specialist:
                class_profile = "bdd10"
            super().__init__(model_path, input_size, specialist=specialist,
                             class_names=class_names, class_profile=class_profile)

        def detect(self, frame):
            result = self.detector.infer(frame, stream_id=0,
                                         frame_id=self.frame_id, timestamp_ns=None)
            self.frame_id += 1
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
                if not self.specialist and not self.single_model and class_name == "traffic light":
                    continue
                if not self.specialist and class_name in ("motorcycle", "bicycle") and detection.confidence < 0.85:
                    continue
                if class_name not in competition_classes:
                    continue
                items.append({
                    "class_name": class_name,
                    "confidence": detection.confidence,
                    "bbox": [detection.x1, detection.y1, detection.x2, detection.y2],
                    "source": self.source,
                })
            return items, result.inference_ms

        def close(self):
            super().close()

    bridge_module.RknnConsoleDetector = CompetitionConsoleDetector


def run_benchmark(args) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    replay_paths = _make_replay_frames(args.data_root, args.replay_dir, args.frame_count)
    main_kind = "p2" if args.scheme in {"B", "D"} else "bdd"
    main_model = args.model_dir / MODEL_FILES[main_kind]
    specialist_model = args.model_dir / MODEL_FILES["specialist"]
    for path in (main_model, specialist_model):
        if not path.is_file():
            raise FileNotFoundError(path)

    os.environ["ADAS_EDGE_ROOT"] = str(args.competition_root)
    sys.path.insert(0, str(args.competition_root))
    sys.path.insert(0, str(args.bridge_script.parent))
    spec = importlib.util.spec_from_file_location("b05p_ai_bridge", args.bridge_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load competition bridge: {args.bridge_script}")
    bridge_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge_module
    spec.loader.exec_module(bridge_module)
    _install_raw_head_adapter(bridge_module, args.scheme)

    request_port = _unused_udp_port()
    result_port = _unused_udp_port()
    output_summary = args.output_dir / "bridge_summary.json"
    timing_log = args.output_dir / "bridge_timing.jsonl"
    argv = [
        "--source", "SYNTHETIC", "--layout", "quad", "--display-view", "front",
        *_replay_quadrant_args(), "--navigation-quadrant", "1",
        "--main-model", str(main_model),
        "--main-hz", str(args.request_hz), "--hazard-hz", "5", "--specialist-hz", "1",
        "--request-port", str(request_port), "--result-port", str(result_port),
        "--idle-timeout", "60", "--summary", str(output_summary),
        "--timing-log", str(timing_log),
    ]
    if args.scheme in {"C", "D"}:
        argv.extend(("--specialist-model", str(specialist_model)))
    for path in replay_paths:
        argv.extend(("--file", str(path)))
    bridge_args = bridge_module.parser().parse_args(argv)
    bridge = bridge_module.AiReplayBridge(bridge_args)
    bridge.telemetry.start()
    bridge.start()
    time.sleep(0.2)

    request_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    total_requests = args.warmup + args.requests
    start_ns = time.monotonic_ns()
    try:
        for index in range(total_requests):
            deadline_ns = start_ns + int(index * 1e9 / args.request_hz)
            while True:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                time.sleep(min(remaining_ns / 1e9, 0.005))
            timestamp_ns = time.monotonic_ns()
            request = bridge_module.encode_frame_request(index, timestamp_ns)
            request_socket.sendto(request, ("127.0.0.1", request_port))
    finally:
        request_socket.close()

    drain_deadline = time.monotonic() + 10.0
    while time.monotonic() < drain_deadline:
        if bridge.request_count >= total_requests and bridge.result_count + bridge.replaced_count >= total_requests:
            break
        time.sleep(0.05)
    bridge.close()

    summary = json.loads(output_summary.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in timing_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    measured = [row for row in rows if int(row["frame_id"]) >= args.warmup]
    measured_seconds = args.requests / args.request_hz
    report = {
        "schema": "b05p-competition-bridge-runtime-v1",
        "scheme": args.scheme,
        "source": "Cityscapes replay through the board's actual AiReplayBridge request, scheduler, detector adapter, result mapping and UDP publication path; PCIe capture and Qt display excluded.",
        "models": {
            "main": {"path": str(main_model), "sha256": _sha256(main_model)},
            "specialist": ({"path": str(specialist_model), "sha256": _sha256(specialist_model)}
                            if args.scheme in {"C", "D"} else None),
        },
        "input": {
            "frame_format": (
                "1280x720 BGR565 packed replay, road image in quadrant "
                f"{REPLAY_ROAD_QUADRANT}"
            ),
            "road_quadrant": REPLAY_ROAD_QUADRANT,
            "road_crop": "640x360",
            "requests_hz": args.request_hz,
            "warmup_requests": args.warmup,
            "measured_requests": args.requests,
            "replay_frames": len(replay_paths),
            "fusion": "single model keeps main traffic-light detections; dual models replace main traffic-light detections with specialist results, matching the live bridge adapter.",
        },
        "measurement": {
            "measured_frames_analyzed": len(measured),
            "measured_result_fps": len(measured) / measured_seconds,
            "measured_input_span_seconds": measured_seconds,
            "all_requests_received": summary["input_requests_received"],
            "all_results_published": summary["result_packets_sent"],
            "requests_replaced_latest_only": summary["source_ids_replaced_before_analysis"],
            "inference_errors": summary["inference_errors"],
            "main_model_timing": _stats([
                row["main_timing"]["total_ms"] for row in measured if row.get("main_timing")
            ]),
            "main_npu_timing": _stats([
                row["main_timing"]["rknn_call_ms"] for row in measured if row.get("main_timing")
            ]),
            "main_decode_nms_timing": _stats([
                row["main_timing"]["decode_nms_ms"] for row in measured if row.get("main_timing")
            ]),
            "bridge_pipeline_timing": _stats([
                row["pipeline_ms"] for row in measured if row.get("pipeline_ms") is not None
            ]),
            "result_latency_timing": _stats([
                (row["result_publish_timestamp_ns"] - row["ai_receive_timestamp_ns"]) / 1e6
                for row in measured
                if row.get("result_publish_timestamp_ns") is not None
            ]),
            "specialist_model_timing": _stats([
                row["specialist_timing"]["total_ms"] for row in measured if row.get("specialist_timing")
            ]),
            "bridge_summary": summary,
        },
    }
    (args.output_dir / "runtime_measurement.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--bridge-script", type=Path, required=True)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--request-hz", type=float, default=10.0)
    args = parser.parse_args()
    if args.frame_count <= 0 or args.warmup < 0 or args.requests <= 0 or args.request_hz <= 0:
        parser.error("frame-count and requests must be positive; warmup non-negative; request-hz positive")
    report = run_benchmark(args)
    print(json.dumps({
        "scheme": report["scheme"],
        "measured_result_fps": report["measurement"]["measured_result_fps"],
        "measured_frames_analyzed": report["measurement"]["measured_frames_analyzed"],
        "requests_replaced_latest_only": report["measurement"]["requests_replaced_latest_only"],
        "main_model_timing": report["measurement"]["main_model_timing"],
        "main_npu_timing": report["measurement"]["main_npu_timing"],
        "main_decode_nms_timing": report["measurement"]["main_decode_nms_timing"],
        "bridge_pipeline_timing": report["measurement"]["bridge_pipeline_timing"],
        "output": str(args.output_dir / "runtime_measurement.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
