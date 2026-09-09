"""Software-only competition scenario replay (never a PCIe hardware claim)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from adas_edge.competition import (
    CompetitionDetector, CompetitionPipeline, FrameDescriptor, LaneState,
    NavigationCommand, PixelFormat, SceneInput, TrafficLightState, TurnSignal,
    classify_traffic_lights,
)
from adas_edge.competition.policy import AdasAction


def _enum(enum_type, value, default):
    try:
        return enum_type(value)
    except (ValueError, TypeError):
        return default


def _event_for(events, frame_id, timestamp):
    for event in events:
        if "start_frame" in event and int(event["start_frame"]) <= frame_id <= int(event.get("end_frame", event["start_frame"])):
            return event
        if "start_timestamp" in event and float(event["start_timestamp"]) <= timestamp <= float(event.get("end_timestamp", event["start_timestamp"])):
            return event
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", required=True, help="JSON scenario event file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--stream-id", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    events = scenario.get("events", [])
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    detection_rows, adas_rows = [], []
    frame_id = 0
    with CompetitionDetector(args.model, args.input_size) as detector:
        pipeline = CompetitionPipeline(detector)
        while cap.isOpened() and (args.max_frames is None or frame_id < args.max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
            timestamp = frame_id / fps
            event = _event_for(events, frame_id, timestamp)
            descriptor = FrameDescriptor(args.stream_id, frame_id, int(timestamp * 1_000_000_000), frame.shape[1], frame.shape[0], PixelFormat.BGR888, frame.shape[1] * 3, frame.tobytes())
            detections = detector.detect(frame)
            lights = [d for d in detections if d.class_name == "traffic light"]
            light = _enum(TrafficLightState, event.get("traffic_light_state"), classify_traffic_lights(frame, lights) if lights else TrafficLightState.UNKNOWN)
            scene = SceneInput(
                traffic_light_state=light,
                lane_state=_enum(LaneState, event.get("lane_type"), LaneState.UNKNOWN),
                navigation_command=_enum(NavigationCommand, event.get("navigation_command"), NavigationCommand.NONE),
                requested_maneuver=_enum(NavigationCommand, event.get("requested_maneuver"), NavigationCommand.NONE),
                turn_signal=_enum(TurnSignal, event.get("turn_signal"), TurnSignal.NONE),
                current_speed_kph=event.get("current_speed"), speed_limit_kph=event.get("speed_limit"),
                pedestrian_hazard=bool(event.get("pedestrian_hazard", any(d.class_name == "person" and d.bbox_xyxy[3] >= frame.shape[0] * .8 for d in detections))),
                obstacle_hazard=bool(event.get("obstacle_hazard", any(d.class_name in {"car", "bus", "truck", "motorcycle", "bicycle"} and d.bbox_xyxy[3] >= frame.shape[0] * .8 for d in detections))),
                forced_red_light_violation=bool(event.get("forced_red_light_violation", False)),
            )
            result = pipeline.process_descriptor(descriptor, scene)
            for record in result.detections:
                detection_rows.append({"stream_id": args.stream_id, "frame_id": frame_id, "timestamp": timestamp, "class_id": record.class_id, "confidence": record.confidence, "x1": record.x1, "y1": record.y1, "x2": record.x2, "y2": record.y2})
                cv2.rectangle(frame, (record.x1, record.y1), (record.x2, record.y2), (0, 255, 0), 2)
            adas_rows.append({"stream_id": args.stream_id, "frame_id": frame_id, "timestamp": timestamp, "traffic_light_state": scene.traffic_light_state.name, "lane_state": scene.lane_state.name, "navigation_command": scene.navigation_command.value, "requested_maneuver": scene.requested_maneuver.value, "turn_signal": scene.turn_signal.value, "speed": scene.current_speed_kph, "speed_limit": scene.speed_limit_kph, "adas_action": AdasAction(result.adas_action).name, "warning_flags": int(result.warning_flags)})
            cv2.putText(frame, f"{AdasAction(result.adas_action).name} {int(result.warning_flags)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255) if result.adas_action == AdasAction.STOP else (0, 255, 0), 2)
            writer.write(frame)
            frame_id += 1
    cap.release()
    if writer is not None:
        writer.release()
    def write_rows(name, rows):
        fields = list(rows[0]) if rows else []
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer_ = csv.DictWriter(handle, fieldnames=fields); writer_.writeheader(); writer_.writerows(rows)
    write_rows("detections.csv", detection_rows)
    write_rows("adas_log.csv", adas_rows)
    summary = {"status": "SOFTWARE E2E MOCK", "frames": frame_id, "detections": len(detection_rows), "scenario": str(args.scenario)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
