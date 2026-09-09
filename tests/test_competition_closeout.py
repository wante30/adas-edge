import json
from pathlib import Path

import numpy as np
import pytest
import cv2

from adas_edge.competition.pipeline import CompetitionPipeline
from adas_edge.competition.pixel import PixelFormat, bgr_to_payload, payload_to_bgr
from adas_edge.competition.policy import (
    AdasAction,
    CompetitionPolicy,
    LaneState,
    NavigationCommand,
    SceneInput,
    TrafficLightState,
    TurnSignal,
    WarningFlags,
)
from adas_edge.competition.protocol import (
    DetectionRecord,
    FrameDescriptor,
    FLAG_CRC32,
    ProtocolError,
    ResultDescriptor,
)
from adas_edge.contracts import Detection


def test_protocol_matches_golden_vectors_and_round_trips():
    vectors = json.loads((Path(__file__).parent / "golden" / "pcie_v1_vectors.json").read_text())
    frame = FrameDescriptor(
        stream_id=2, frame_id=8, timestamp_ns=99, width=1, height=1,
        pixel_format=PixelFormat.BGR888, stride=3, payload=b"abc", flags=0,
    )
    result = ResultDescriptor(
        stream_id=2, frame_id=8, timestamp_ns=99, num_objects=1,
        traffic_light_state=TrafficLightState.RED, lane_state=LaneState.DASHED,
        adas_action=AdasAction.STOP, warning_flags=WarningFlags.RED_LIGHT,
        detections=(DetectionRecord(2, 24576, 1, 2, 3, 4),),
    )
    assert frame.serialize().hex() == vectors["frame"].replace(" ", "")
    assert result.serialize().hex() == vectors["result"].replace(" ", "")
    assert FrameDescriptor.deserialize(frame.serialize()) == frame
    assert ResultDescriptor.deserialize(result.serialize()) == result


@pytest.mark.parametrize("packet", [b"XXXX" + b"\0" * 60, b"ADSF" + b"\0" * 4])
def test_protocol_rejects_malformed_headers(packet):
    with pytest.raises(ProtocolError):
        FrameDescriptor.deserialize(packet)


def test_protocol_rejects_version_size_and_crc_errors():
    frame = FrameDescriptor(0, 1, 2, 2, 1, PixelFormat.BGR888, 6, b"abcdef")
    frame = FrameDescriptor(0, 1, 2, 2, 1, PixelFormat.BGR888, 6, b"abcdef", FLAG_CRC32)
    packet = bytearray(frame.serialize())
    packet[4:6] = (99).to_bytes(2, "little")
    with pytest.raises(ProtocolError, match="version"):
        FrameDescriptor.deserialize(bytes(packet))
    packet = bytearray(frame.serialize())
    packet[-1] ^= 1
    with pytest.raises(ProtocolError):
        FrameDescriptor.deserialize(bytes(packet))
    result = ResultDescriptor(0, 1, 2, 0, TrafficLightState.GREEN, LaneState.UNKNOWN, AdasAction.FORWARD, WarningFlags.NONE, ())
    bad_result = bytearray(result.serialize())
    bad_result[0:4] = b"NOPE"
    with pytest.raises(ProtocolError):
        ResultDescriptor.deserialize(bytes(bad_result))
    bad_result = bytearray(result.serialize())
    bad_result[4:6] = (99).to_bytes(2, "little")
    with pytest.raises(ProtocolError, match="version"):
        ResultDescriptor.deserialize(bytes(bad_result))


def test_pixel_adapters_cover_all_wire_formats():
    bgr = np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
    assert np.array_equal(payload_to_bgr(bgr.tobytes(), 3, 1, PixelFormat.BGR888, 9), bgr)
    assert np.array_equal(payload_to_bgr(bgr[:, :, ::-1].tobytes(), 3, 1, PixelFormat.RGB888, 9), bgr)
    bgr_nv12 = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr_nv12[0, 0] = (0, 0, 255)
    i420 = cv2.cvtColor(bgr_nv12, cv2.COLOR_BGR2YUV_I420)
    nv12 = np.vstack((i420[:2], np.stack((i420[2, :1], i420[2, 1:2]), axis=1)))
    assert payload_to_bgr(nv12.tobytes(), 2, 2, PixelFormat.NV12, 2).shape == bgr_nv12.shape
    assert len(bgr_to_payload(bgr_nv12, PixelFormat.NV12)) == 6
    for color in [(0, 0, 255), (0, 255, 0), (255, 0, 0)]:
        payload = bgr_to_payload(np.array([[color]], dtype=np.uint8), PixelFormat.RGB565)
        assert tuple(payload_to_bgr(payload, 1, 1, PixelFormat.RGB565, 2)[0, 0]) == color


class StubDetector:
    def detect(self, frame):
        return [
            Detection(0, "person", 0.9, (-4, 2, 20, 40)),
            Detection(2, "car", 0.8, (90, 90, 20, 10)),
        ]


def test_pipeline_preserves_identity_and_clips_detections():
    frame = FrameDescriptor(3, 17, 123, 4, 2, PixelFormat.BGR888, 12, bytes(24))
    result = CompetitionPipeline(StubDetector()).process_descriptor(
        frame, SceneInput(traffic_light_state=TrafficLightState.GREEN)
    )
    assert (result.stream_id, result.frame_id, result.timestamp_ns) == (3, 17, 123)
    assert len(result.detections) == 1
    assert result.detections[0].x1 == 0 and result.detections[0].x2 == 3


@pytest.mark.parametrize(
    "scene, action, flag",
    [
        (SceneInput(traffic_light_state=TrafficLightState.RED), AdasAction.STOP, WarningFlags.RED_LIGHT),
        (SceneInput(traffic_light_state=TrafficLightState.YELLOW), AdasAction.STOP, WarningFlags.YELLOW_LIGHT),
        (SceneInput(traffic_light_state=TrafficLightState.UNKNOWN), AdasAction.STOP, WarningFlags.TRAFFIC_LIGHT_UNKNOWN),
        (SceneInput(pedestrian_hazard=True), AdasAction.STOP, WarningFlags.PEDESTRIAN_DANGER),
        (SceneInput(obstacle_hazard=True), AdasAction.STOP, WarningFlags.COLLISION_WARNING),
        (SceneInput(traffic_light_state=TrafficLightState.GREEN, lane_state=LaneState.DASHED, requested_maneuver=NavigationCommand.CHANGE_LEFT, turn_signal=TurnSignal.LEFT), AdasAction.LEFT, WarningFlags.NONE),
        (SceneInput(traffic_light_state=TrafficLightState.GREEN, lane_state=LaneState.SOLID, requested_maneuver=NavigationCommand.CHANGE_LEFT, turn_signal=TurnSignal.LEFT), AdasAction.FORWARD, WarningFlags.SOLID_LANE_CHANGE_FORBIDDEN),
        (SceneInput(traffic_light_state=TrafficLightState.GREEN, current_speed_kph=70, speed_limit_kph=50), AdasAction.FORWARD, WarningFlags.OVERSPEED),
    ],
)
def test_policy_scenarios(scene, action, flag):
    result = CompetitionPolicy().decide(scene)
    assert result.action == action
    assert flag == WarningFlags.NONE or flag in result.warning_flags


def test_temporal_stabilizer_handles_unknown_and_transition():
    from adas_edge.competition.traffic_light import TrafficLightStabilizer

    tracker = TrafficLightStabilizer(window_size=3, green_confirmations=2, stale_timeout=2)
    assert tracker.update(TrafficLightState.GREEN) == TrafficLightState.UNKNOWN
    assert tracker.update(TrafficLightState.GREEN) == TrafficLightState.GREEN
    assert tracker.update(TrafficLightState.UNKNOWN) == TrafficLightState.GREEN
    assert tracker.update(TrafficLightState.RED) == TrafficLightState.RED


def test_four_stream_transport_preserves_identity_and_drop_counts():
    from adas_edge.competition.transport import RoundRobinTransport

    transport = RoundRobinTransport(streams=4, queue_depth=2)
    for stream_id in range(4):
        for frame_id in range(3):
            transport.push(FrameDescriptor(stream_id, frame_id, 0, 1, 1, PixelFormat.BGR888, 3, b"\0" * 3))
    seen = []
    while (frame := transport.pop()) is not None:
        seen.append((frame.stream_id, frame.frame_id))
    assert [sid for sid, _ in seen[:4]] == [0, 1, 2, 3]
    assert transport.dropped == [1, 1, 1, 1]
