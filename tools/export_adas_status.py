"""Export existing video inference logs as ADST; this does not run inference."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from competition_io.adas_status_v1 import (Action, AdasReceiver, AdasStatus, ClockDomain,
                                         Light, SIZE, Valid, Warning)


def export_log(source: Path, output: Path, ttl_ms: int = 750) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if not 1 <= ttl_ms <= 60000:
        raise ValueError("ttl_ms outside ADST range")
    now = [0]
    receiver = AdasReceiver(ClockDomain.VIDEO_REPLAY, lambda: now[0])
    colours, actions, reasons = Counter(), Counter(), Counter()
    count, last_frame, end_ns = 0, -1, 0
    last_observation = None
    temporal_holds = 0
    with source.open(encoding="utf-8") as records, (output / "official_video_adst.bin").open("wb") as binary, \
            (output / "official_video_adst.jsonl").open("w", encoding="utf-8") as readable:
        for line in records:
            row = json.loads(line)
            frame, current_ns = int(row["frame"]), round(float(row["source_s"])*1_000_000_000)
            if frame <= last_frame or current_ns < end_ns:
                raise ValueError("Log must be a single monotonically ordered video replay")
            last_frame, end_ns = frame, current_ns
            if not row["main_fresh"]:
                continue
            signal = row["signal"]
            light = Light[signal["state"].upper()]
            # These particular logs contain no verified ego-signal association
            # or lane detections. Keep both decisions and lane fields UNKNOWN.
            main_age = float(row["main_age_ms"])
            source_frame = int(row["main_source_frame"])
            source_age = main_age
            confidence = float(signal["confidence"])
            timestamp_ns = max(0, current_ns-round(source_age*1_000_000))
            held = False
            if light != Light.UNKNOWN:
                # Aggregated signal metadata does not identify its selected
                # detector. Bound its age by the older contributing stream.
                source_age = max(main_age, float(row["specialist_age_ms"]))
                source_frame = min(source_frame, int(row["specialist_source_frame"]))
                timestamp_ns = max(0, current_ns-round(source_age*1_000_000))
                if signal["raw_state"].upper() == light.name:
                    last_observation = (light, source_frame, timestamp_ns, confidence)
                elif last_observation and last_observation[0] == light:
                    # Temporal smoothing held an earlier colour or has not
                    # confirmed a new colour. Preserve that earlier evidence.
                    light, source_frame, timestamp_ns, confidence = last_observation
                    source_age = (current_ns-timestamp_ns)/1_000_000
                    held = True
                    temporal_holds += 1
                else:
                    light, confidence = Light.UNKNOWN, 0.
            else:
                last_observation = None
            valid = Valid.SIGNAL_OBSERVED if light != Light.UNKNOWN else Valid(0)
            warnings = Warning.UNVERIFIED_ASSOCIATION | Warning.UNKNOWN_LANE
            if light == Light.UNKNOWN:
                warnings |= Warning.UNKNOWN_SIGNAL
            count += 1
            message = AdasStatus(count, 0, source_frame, timestamp_ns, light,
                                 signal_confidence=confidence, valid=valid,
                                 warnings=warnings, valid_for_ms=ttl_ms)
            packet = message.serialize()
            assert AdasStatus.deserialize(packet).serialize() == packet
            now[0] = current_ns
            view = receiver.receive(packet)
            assert view.action == Action.UNKNOWN  # never fabricate GO from this log
            binary.write(packet)
            colours[light.name] += 1
            actions[view.action.name] += 1
            reasons[view.reason] += 1
            readable.write(json.dumps({"video_frame": frame, "video_timestamp_ns": current_ns,
                                       "conservative_source_age_ms": source_age,
                                       "temporal_hold_preserved_source_time": held,
                                       "status": asdict(message), "receiver": asdict(view)}, ensure_ascii=False) + "\n")
    if not count:
        raise ValueError("No main inference updates found")
    now[0] = end_ns + ttl_ms*1_000_000
    expired = receiver.current()
    assert expired.action == Action.UNKNOWN and expired.warnings & Warning.STALE
    summary = {"evidence": "PC_REPLAY_OF_EXISTING_UNLABELLED_VIDEO_INFERENCE_LOG_NOT_BOARD_OR_ACCURACY_TEST",
               "passed": True, "source_log": str(source.resolve()),
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "video_frame_count": last_frame+1, "packet_count": count, "packet_size": SIZE,
               "binary_size": (output / "official_video_adst.bin").stat().st_size,
               "stream_id": 0, "clock_domain": "VIDEO_REPLAY", "valid_for_ms": ttl_ms,
               "light_observation_packet_counts": dict(colours), "receiver_action_counts": dict(actions),
               "receiver_reason_counts": dict(reasons), "expiry_verified": True,
               "temporal_hold_packets_preserving_original_time": temporal_holds,
               "limitations": ["Inference was not rerun; input is the existing PC FP32 ONNX log",
                               "Light colours are experimental observations, not ego-lane permissions",
                               "No lane detections; both lane fields and all actions remain UNKNOWN",
                               "Source time is conservatively bounded by the older contributing detector",
                               "Temporally held colours retain their last matching observation time",
                               "ADST has no assigned BAR address or FPGA consumer"]}
    (output / "adas_export.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ttl-ms", type=int, default=750)
    args = parser.parse_args()
    print(json.dumps(export_log(args.input, args.output, args.ttl_ms), ensure_ascii=False, indent=2))
