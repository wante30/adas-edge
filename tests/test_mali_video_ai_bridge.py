import importlib.util
from pathlib import Path

import numpy as np

from tools.competition_console import runtime


AI_BRIDGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "lubancat"
    / "mali_video"
    / "ai_bridge.py"
)


def _load_ai_bridge():
    assert AI_BRIDGE_PATH.is_file(), "Mali video AI bridge is missing"
    spec = importlib.util.spec_from_file_location("mali_video_ai_bridge_test", AI_BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_frame_slot_replaces_pending_requests_without_queueing():
    slot_type = getattr(runtime, "LatestFrameSlot", None)
    assert slot_type is not None, "latest-only inference request slot is missing"

    slot = slot_type()
    first = (17, 1000)
    newest = (23, 2000)

    assert slot.publish(first) is False
    assert slot.publish(newest) is True
    assert slot.replaced_count == 1
    assert slot.take(timeout=0) == newest
    assert slot.take(timeout=0) is None
    slot.close()


def test_frame_request_wire_format_is_little_endian_and_rejects_bad_packets():
    bridge = _load_ai_bridge()
    payload = bridge.encode_frame_request(0x0102030405060708, 9876543210)

    assert payload[:4] == b"MVR1"
    assert len(payload) == 20
    assert bridge.decode_frame_request(payload) == (0x0102030405060708, 9876543210)
    assert bridge.decode_frame_request(payload[:-1]) is None
    assert bridge.decode_frame_request(b"BAD!" + payload[4:]) is None


def test_rgb565_replay_decode_matches_the_existing_pgv_bgr_contract():
    bridge = _load_ai_bridge()
    red_rgb565 = bytes((0x00, 0xF8)) * (1280 * 720)

    image = bridge.decode_rgb565_payload(red_rgb565)

    assert image.shape == (720, 1280, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [0, 0, 248]


def test_ai_result_packet_round_trips_status_and_boxes():
    bridge = _load_ai_bridge()
    box = (640.0, 360.0, 812.5, 510.0, 1.0, 0.1, 0.2)

    payload = bridge.encode_ai_result_packet(
        frame_id=120,
        timestamp_ns=99887766,
        flags=0x15,
        nav_direction=2,
        nav_distance_m=42.5,
        boxes=[box],
    )
    result = bridge.decode_ai_result_packet(payload)

    assert result["frame_id"] == 120
    assert result["timestamp_ns"] == 99887766
    assert result["flags"] == 0x15
    assert result["nav_direction"] == 2
    assert result["nav_distance_m"] == 42.5
    assert np.allclose(result["boxes"], [box])
    assert bridge.decode_ai_result_packet(payload[:-1]) is None


def test_board_telemetry_reports_load_frequency_and_temperature_distributions():
    bridge = _load_ai_bridge()
    telemetry = bridge.BoardTelemetry()
    telemetry.samples["npu_load_percent"] = [42.0, 80.0]
    telemetry.samples["npu_frequency_mhz"] = [400.0, 600.0]
    telemetry.zones = [("soc-thermal", Path("/unused"))]
    telemetry.temperatures = [
        {"soc-thermal": [29.0]},
        {"soc-thermal": [35.0]},
    ]

    summary = telemetry.summary()

    assert summary["devices"]["npu"]["load_percent"]["p95_percent"] == 78.1
    assert summary["devices"]["npu"]["frequency_mhz"]["p50_mhz"] == 500.0
    assert summary["temperatures_c"]["soc-thermal"]["max_c"] == 35.0
