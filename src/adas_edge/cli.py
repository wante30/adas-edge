from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2

from adas_edge.benchmark import BenchmarkRecorder
from adas_edge.perception.ultralytics_detector import UltralyticsDetector
from adas_edge.pipeline import AdasPipeline
from adas_edge.render import draw_result
from adas_edge.transport.control import ConsoleControlTransport, UDPControlTransport
from adas_edge.video.local import LocalFileSource
from adas_edge.video.usb import USBCameraSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transport-independent Phase-1 ADAS baseline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Image, video file, or directory of images")
    source.add_argument("--camera", type=int, help="USB camera index")
    parser.add_argument("--camera-id", default="front")
    parser.add_argument("--model", default="yolo11n.pt", help="COCO YOLO weights path or Ultralytics name")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--show", action="store_true", help="Show annotated frames in a local window")
    parser.add_argument("--output-video", type=Path, help="Optional annotated mp4 output")
    parser.add_argument("--output-jsonl", type=Path, help="Optional per-frame JSON Lines output")
    parser.add_argument("--benchmark-csv", type=Path, help="Optional per-frame benchmark CSV")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means read until source exhaustion")
    parser.add_argument("--udp-control", help="Optional HOST:PORT control output; console is the default")
    return parser


def _make_source(args: argparse.Namespace):
    if args.source:
        return LocalFileSource(args.source, camera_id=args.camera_id)
    return USBCameraSource(args.camera, camera_id=args.camera_id)


def _make_control_transport(args: argparse.Namespace):
    if not args.udp_control:
        return ConsoleControlTransport(sys.stderr)
    host, port = args.udp_control.rsplit(":", 1)
    return UDPControlTransport(host, int(port))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = _make_source(args)
    control = _make_control_transport(args)
    detector = UltralyticsDetector(args.model, args.confidence, args.input_size, args.device)
    pipeline = AdasPipeline(detector)
    recorder = BenchmarkRecorder(detector.model_name, args.input_size, "FP32")
    output_fps = 20.0
    if args.source and Path(args.source).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        probe = cv2.VideoCapture(str(args.source))
        if probe.isOpened():
            probed_fps = probe.get(cv2.CAP_PROP_FPS)
            if probed_fps and probed_fps > 0:
                output_fps = probed_fps
        probe.release()
    writer: cv2.VideoWriter | None = None
    results_file = None
    processed = 0
    started = perf_counter()

    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        results_file = args.output_jsonl.open("w", encoding="utf-8")
    try:
        while args.max_frames == 0 or processed < args.max_frames:
            frame = source.read()
            if frame is None:
                break
            result = pipeline.process(frame)
            control.send(result.command)
            payload = result.to_dict()
            payload["stream_fps"] = round((processed + 1) / (perf_counter() - started), 3)
            line = json.dumps(payload, ensure_ascii=False)
            print(line)
            if results_file:
                results_file.write(line + "\n")
            if args.benchmark_csv:
                BenchmarkRecorder.append_csv(args.benchmark_csv, recorder.row(result))

            rendered = draw_result(result)
            cv2.putText(
                rendered,
                f"FPS: {payload['stream_fps']:.1f}",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            if args.output_video:
                if writer is None:
                    args.output_video.parent.mkdir(parents=True, exist_ok=True)
                    height, width = rendered.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                writer.write(rendered)
            if args.show:
                cv2.imshow("ADAS Edge", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            processed += 1
    finally:
        source.close()
        if hasattr(control, "close"):
            control.close()
        if writer is not None:
            writer.release()
        if results_file:
            results_file.close()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = perf_counter() - started
    print(json.dumps({"summary": {"frames": processed, "elapsed_s": round(elapsed, 3), "fps": round(processed / elapsed, 3) if elapsed else 0.0}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
