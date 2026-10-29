import importlib.util
from pathlib import Path
import threading
import sys
from types import SimpleNamespace

import numpy as np

from tools.competition_console import runtime


AI_BRIDGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "lubancat"
    / "mali_video"
    / "ai_bridge.py"
)
QT_AI_BRIDGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "lubancat"
    / "qt_gpu_probe"
    / "ai_bridge.py"
)
B05P_ADAPTER_PATH = QT_AI_BRIDGE_PATH.with_name("b05p_competition_bridge.py")


def _load_ai_bridge():
    assert AI_BRIDGE_PATH.is_file(), "Mali video AI bridge is missing"
    spec = importlib.util.spec_from_file_location("mali_video_ai_bridge_test", AI_BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_qt_ai_bridge():
    assert QT_AI_BRIDGE_PATH.is_file(), "Qt GPU probe AI bridge is missing"
    sys.path.insert(0, str(QT_AI_BRIDGE_PATH.parent))
    spec = importlib.util.spec_from_file_location("qt_ai_bridge_test", QT_AI_BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_qt_ai_bridge_reuses_identical_road_and_navigation_quadrants(monkeypatch):
    bridge = _load_qt_ai_bridge()
    payload = bytes(bridge.FRAME_BYTES)
    packed = bridge.PackedRgb565(payload)
    calls = []
    road = object()

    def decode(self, quadrant):
        calls.append(quadrant)
        return road if quadrant == 1 else object()

    monkeypatch.setattr(bridge.PackedRgb565, "quadrant_bgr", decode)
    selected_road, selected_navigation = bridge.select_inference_frames(
        packed, "quad", 1, 1
    )

    assert calls == [1]
    assert selected_road is selected_navigation is road


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


def test_frame_request_v2_carries_renderer_measured_fps():
    bridge = _load_ai_bridge()

    payload = bridge.encode_frame_request(
        42, 123456789, input_fps=59.85, submitted_fps=59.68
    )

    assert payload[:4] == b"MVR2"
    decoded = bridge.decode_frame_request(payload)
    assert decoded[:2] == (42, 123456789)
    assert np.allclose(decoded[2:], (59.85, 59.68), atol=0.01)


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


def test_ai_result_v2_preserves_class_confidence_and_dynamic_status():
    bridge = _load_ai_bridge()
    status = {
        "source": "SYNTHETIC",
        "traffic": "UNKNOWN",
        "navigation": {"state": "UNREADABLE", "road_name": ""},
        "hazards": ["PEDESTRIAN_CROSSING_PATH"],
    }
    box = (10.0, 20.0, 30.0, 60.0, 1.0, 0.2, 0.1, 0, 0.875)

    payload = bridge.encode_ai_result_packet(
        frame_id=42,
        timestamp_ns=123456789,
        flags=0,
        nav_direction=0,
        nav_distance_m=-1.0,
        boxes=[box],
        status=status,
    )
    result = bridge.decode_ai_result_packet(payload)

    assert payload[:4] == b"MVA2"
    assert result["frame_id"] == 42
    assert result["boxes"][0][7:] == (0, 0.875)
    assert result["status"] == status


def test_detection_box_mapping_keeps_quad_and_front_coordinates_distinct():
    bridge = _load_ai_bridge()

    assert bridge.map_detection_box(
        (10, 20, 50, 80), input_layout="quad", display_view="quad", road_quadrant=3
    ) == (650.0, 380.0, 690.0, 440.0)
    assert bridge.map_detection_box(
        (10, 20, 50, 80), input_layout="quad", display_view="front", road_quadrant=3
    ) == (20.0, 40.0, 100.0, 160.0)
    assert bridge.map_detection_box(
        (10, 20, 50, 80), input_layout="front", display_view="front", road_quadrant=3
    ) == (10.0, 20.0, 50.0, 80.0)


def test_dynamic_status_expires_navigation_and_never_carries_old_road_name():
    bridge = _load_ai_bridge()
    now_ns = 2_000_000_000
    status = bridge.build_dynamic_status(
        source="LOCAL_REPLAY",
        frame_id=120,
        timestamp_ns=now_ns,
        now_ns=now_ns,
        hud={"traffic": "OFF", "traffic_evidence": "YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL"},
        navigation={
            "valid": True,
            "direction": "left",
            "distance_m": 42,
            "road_name": "上一条道路",
            "source_frame_id": 90,
            "timestamp_ns": now_ns - 800_000_000,
            "reason": "PIXEL_ARROW_AND_OCR_DISTANCE_VIDEO_LAYOUT_V1",
        },
        hazard_timestamp_ns=now_ns - 50_000_000,
        hazards=[{"kind": "LEAD_RAPID_CLOSING", "track_id": 7}],
        detections=[{"class_name": "person", "confidence": 0.91}],
        signal_age_ms=30.0,
        main_age_ms=40.0,
        specialist_age_ms=80.0,
    )

    assert status["traffic"] == "YELLOW"
    assert status["navigation"] == {
        "state": "STOPPED",
        "direction": "none",
        "distance_m": None,
        "road_name": "",
        "source_frame_id": 90,
        "age_ms": 800.0,
    }
    assert status["hazards"] == ["LEAD_RAPID_CLOSING"]
    assert status["detections"]["person"] == 1
    assert status["source"] == "LOCAL_REPLAY"


def test_dynamic_status_preserves_hazard_target_and_severity_for_mva2():
    bridge = _load_ai_bridge()
    now_ns = 2_000_000_000
    status = bridge.build_dynamic_status(
        source="LOCAL_REPLAY",
        frame_id=120,
        timestamp_ns=now_ns,
        now_ns=now_ns,
        hud={"traffic": "OFF", "traffic_evidence": "NO_SIGNAL_ROI"},
        navigation={},
        hazard_timestamp_ns=now_ns,
        hazards=[{
            "kind": "FRONT_COLLISION_RISK", "track_id": 7,
            "bbox": [100, 80, 180, 150], "target_class": "car",
            "severity": "critical", "source": "IMAGE_TRAJECTORY_EXPERIMENTAL",
        }],
        detections=[{"class_name": "car", "confidence": .9}],
        signal_age_ms=None, main_age_ms=10.0, specialist_age_ms=None,
    )
    assert status["hazard_details"] == [{
        "event_type": "FRONT_COLLISION_RISK",
        "source_id": "LOCAL_REPLAY",
        "source_frame_id": 120,
        "timestamp": now_ns,
        "severity": "critical",
        "valid": True,
        "ttl_ms": 250.0,
        "evidence": "IMAGE_TRAJECTORY_EXPERIMENTAL",
        "track_id": 7,
        "bbox": [100.0, 80.0, 180.0, 150.0],
        "target_class": "car",
    }]


def test_hazard_analysis_has_explicit_disable_flag():
    bridge_module = _load_qt_ai_bridge()

    args = bridge_module.parser().parse_args([
        "--live-frame-file", "/tmp/frame.mfr1",
        "--main-model", "/tmp/model.rknn",
        "--no-hazards",
    ])

    assert args.hazards is False


def test_dynamic_status_expires_detection_counts_with_the_drawn_box_ttl():
    bridge = _load_ai_bridge()
    now_ns = 2_000_000_000
    status = bridge.build_dynamic_status(
        source="LOCAL_REPLAY",
        frame_id=120,
        timestamp_ns=now_ns - 251_000_000,
        now_ns=now_ns,
        hud={"traffic": "OFF", "traffic_evidence": "NO_SIGNAL_ROI"},
        navigation={},
        hazard_timestamp_ns=None,
        hazards=(),
        detections=[{"class_name": "person"}],
        signal_age_ms=None,
        main_age_ms=100.0,
        specialist_age_ms=None,
    )

    assert status["result_age_ms"] == 251.0
    assert status["detections"] == {}


def test_dynamic_status_clears_live_results_when_source_stops():
    bridge = _load_ai_bridge()
    now_ns = 4_000_000_000
    status = bridge.build_dynamic_status(
        source="PGV_LIVE",
        frame_id=81,
        timestamp_ns=now_ns - 100_000_000,
        now_ns=now_ns,
        hud={"traffic": "RED", "traffic_evidence": "CONFIRMED_RED"},
        navigation={
            "valid": True,
            "direction": "right",
            "distance_m": 30,
            "road_name": "旧道路",
            "source_frame_id": 80,
            "timestamp_ns": now_ns - 100_000_000,
        },
        hazard_timestamp_ns=now_ns - 100_000_000,
        hazards=[{"kind": "PEDESTRIAN_CROSSING_PATH"}],
        detections=[{"class_name": "person"}],
        signal_age_ms=25.0,
        main_age_ms=80.0,
        specialist_age_ms=120.0,
        input_fps=59.8,
        submitted_fps=59.6,
        ai_fps=9.7,
        source_stopped=True,
    )

    assert status["traffic"] == "UNKNOWN"
    assert status["navigation"]["state"] == "STOPPED"
    assert status["navigation"]["road_name"] == ""
    assert status["hazards"] == []
    assert status["detections"] == {}
    assert status["pcie_state"] == "DISCONNECTED"
    assert status["input_fps"] is None
    assert status["submitted_fps"] is None
    assert status["ai_fps"] is None
    assert status["result_age_ms"] is None


def test_stopped_status_heartbeat_waits_for_quiet_source_then_refreshes_periodically():
    bridge = _load_ai_bridge()
    start_ns = 1_000_000_000
    heartbeat_ns = bridge.DYNAMIC_HMI_HEARTBEAT_NS

    assert not bridge.should_refresh_stopped_status(
        start_ns + heartbeat_ns - 1, start_ns, None
    )
    assert bridge.should_refresh_stopped_status(
        start_ns + heartbeat_ns, start_ns, None
    )
    assert not bridge.should_refresh_stopped_status(
        start_ns + 2 * heartbeat_ns - 1, start_ns, start_ns + heartbeat_ns
    )
    assert bridge.should_refresh_stopped_status(
        start_ns + 2 * heartbeat_ns, start_ns, start_ns + heartbeat_ns
    )
    assert not bridge.should_refresh_stopped_status(
        start_ns + heartbeat_ns + 1,
        start_ns + heartbeat_ns,
        start_ns + heartbeat_ns,
    )


def test_initial_pgv_status_is_unknown_and_reports_unavailable_connection():
    bridge = _load_ai_bridge()

    status = bridge.initial_dynamic_status("PGV_LIVE", 123456789)

    assert status["traffic"] == "UNKNOWN"
    assert status["navigation"]["state"] == "UNREADABLE"
    assert status["hazards"] == []
    assert status["detections"] == {}
    assert status["pcie_state"] == "DISCONNECTED"
    assert status["fpga_perception"] == "UNKNOWN"
    assert status["runtime_status"] == "WAITING_FOR_SOURCE"


def test_dynamic_hmi_layer_is_a_bounded_rgba_texture():
    bridge = _load_ai_bridge()
    pixels = bridge.render_dynamic_hmi_rgba(
        {
            "traffic": "UNKNOWN",
            "traffic_reason": "NO_SIGNAL_ROI",
            "navigation": {"state": "UNREADABLE", "road_name": ""},
            "hazards": [],
            "detections": {},
            "source": "SYNTHETIC",
        }
    )

    assert len(pixels) == bridge.DYNAMIC_HMI_WIDTH * bridge.DYNAMIC_HMI_HEIGHT * 4
    assert any(pixels)
    assert pixels[3] == 255


def test_dynamic_hmi_text_is_clipped_to_the_available_pixel_width():
    bridge = _load_ai_bridge()
    image = bridge.Image.new("RGBA", (bridge.DYNAMIC_HMI_WIDTH, bridge.DYNAMIC_HMI_HEIGHT))
    draw = bridge.ImageDraw.Draw(image)
    font = bridge._font(11)

    text = bridge.fit_hmi_text(draw, "导航状态需要显示的完整道路名称和实时诊断信息", font, 202)

    assert text.endswith("...")
    assert bridge.hmi_text_width(draw, text, font) <= 202


def test_live_ai_frame_snapshot_is_rgb565_and_must_match_requested_identity():
    bridge = _load_ai_bridge()
    pixels = bytes((0x00, 0xF8)) * (1280 * 720)
    packet = bridge.encode_frame_snapshot(7, 1234, pixels)

    assert bridge.decode_frame_snapshot(packet, 7, 1234) == pixels
    assert bridge.decode_frame_snapshot(packet, 8, 1234) is None
    assert bridge.decode_frame_snapshot(packet[:-1], 7, 1234) is None


def test_dynamic_hmi_file_header_carries_latest_source_identity():
    bridge = _load_ai_bridge()
    status = {"frame_id": 17, "timestamp_ns": 991122}
    pixels = bytes(bridge.DYNAMIC_HMI_WIDTH * bridge.DYNAMIC_HMI_HEIGHT * 4)
    payload = bridge.encode_dynamic_hmi_file(status, pixels)

    magic, width, height, frame_id, timestamp_ns = bridge.DYNAMIC_HMI_HEADER.unpack_from(payload)
    assert (magic, width, height, frame_id, timestamp_ns) == (b"HMI1", 238, 452, 17, 991122)
    assert len(payload) == bridge.DYNAMIC_HMI_HEADER.size + len(pixels)


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


def _make_switch_bridge(bridge_module, tmp_path, fail_specialist=False, same_main=False):
    sys.path.insert(0, str(QT_AI_BRIDGE_PATH.parent))
    try:
        from model_profiles import ModelProfile
    finally:
        sys.path.pop(0)

    old_main_path = tmp_path / "old-main.rknn"
    old_main_path.write_bytes(b"old")
    new_main_path = tmp_path / "new-main.rknn"
    new_main_path.write_bytes(b"new")
    new_specialist_path = tmp_path / "new-specialist.rknn"
    new_specialist_path.write_bytes(b"new specialist")
    created = []

    class FakeDetector:
        def __init__(self, path, input_size, *, specialist=False, class_profile=None,
                     class_names=None):
            if fail_specialist and specialist:
                raise RuntimeError("specialist load failed")
            self.path = Path(path)
            self.input_size = input_size
            self.specialist = specialist
            self.single_model = not specialist
            self.class_profile = class_profile
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    old_main = FakeDetector(old_main_path, (512, 288), class_profile="bdd10")
    profile = ModelProfile(
        "D", "D test", old_main_path if same_main else new_main_path,
        new_specialist_path, (512, 288), "bdd10"
    )
    bridge_module.RknnConsoleDetector = FakeDetector
    instance = object.__new__(bridge_module.AiReplayBridge)
    instance.args = SimpleNamespace(
        main_model=str(old_main_path), specialist_model="",
        main_hz=10.0, specialist_hz=1.0,
    )
    instance.pipeline = SimpleNamespace(
        main_detector=old_main,
        specialist_detector=None,
        reset=lambda: setattr(instance, "pipeline_was_reset", True),
    )
    instance.model_profiles = {"D": profile}
    instance.model_switch_lock = threading.Lock()
    instance.pending_model_switch = {"request_id": 11, "profile_id": "D"}
    instance.model_switch_peer = None
    instance.active_model_id = "A"
    instance.model_switch_state = "SWITCHING"
    instance.model_switch_message = "loading"
    return instance, old_main, created


def test_runtime_model_switch_replaces_both_detectors_after_success(tmp_path):
    bridge_module = _load_qt_ai_bridge()
    instance, old_main, created = _make_switch_bridge(bridge_module, tmp_path)

    assert instance._apply_pending_model_switch()

    assert instance.active_model_id == "D", instance.model_switch_message
    assert instance.args.main_model.endswith("new-main.rknn")
    assert instance.args.specialist_model.endswith("new-specialist.rknn")
    assert instance.pipeline.main_detector is created[1]
    assert instance.pipeline.main_detector.single_model is False
    assert instance.pipeline.specialist_detector is created[2]
    assert old_main.closed
    assert instance.pipeline_was_reset
    assert instance.model_switch_state == "APPLIED"


def test_runtime_model_switch_keeps_current_pair_when_specialist_load_fails(tmp_path):
    bridge_module = _load_qt_ai_bridge()
    instance, old_main, created = _make_switch_bridge(
        bridge_module, tmp_path, fail_specialist=True
    )

    assert instance._apply_pending_model_switch()

    assert instance.active_model_id == "A"
    assert instance.pipeline.main_detector is old_main
    assert instance.pipeline.specialist_detector is None
    assert instance.args.main_model.endswith("old-main.rknn")
    assert created[1].closed
    assert instance.model_switch_state == "FAILED"


def test_failed_dual_switch_keeps_single_model_filter_when_main_is_reused(tmp_path):
    bridge_module = _load_qt_ai_bridge()
    instance, old_main, created = _make_switch_bridge(
        bridge_module, tmp_path, fail_specialist=True, same_main=True
    )

    assert instance._apply_pending_model_switch()

    assert instance.active_model_id == "A"
    assert instance.pipeline.main_detector is old_main
    assert old_main.single_model is True
    assert instance.pipeline.specialist_detector is None
    assert created == [old_main]
    assert instance.model_switch_state == "FAILED"


def test_b05p_adapter_follows_runtime_single_or_dual_model_state():
    import competition_ai.detector as detector_module

    spec = importlib.util.spec_from_file_location("b05p_adapter_test", B05P_ADAPTER_PATH)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    class FakeRunner:
        def infer(self, frame, stream_id=0, frame_id=0, timestamp_ns=None):
            detections = [
                SimpleNamespace(class_name="traffic light", confidence=0.9,
                                x1=1.0, y1=2.0, x2=5.0, y2=8.0),
                SimpleNamespace(class_name="car", confidence=0.8,
                                x1=10.0, y1=20.0, x2=30.0, y2=40.0),
            ]
            return SimpleNamespace(detections=detections, inference_ms=1.0)

    class FakeBaseDetector:
        def __init__(self, model_path, input_size, **kwargs):
            self.detector = FakeRunner()
            self.model_path = model_path
            self.input_size = input_size

        def close(self):
            pass

    bridge = SimpleNamespace(RknnConsoleDetector=FakeBaseDetector)
    original_detector = detector_module.CompetitionDetector
    try:
        adapter._install_raw_head_adapter(bridge, "A")
        main = bridge.RknnConsoleDetector("bdd.rknn", (512, 288), class_profile="bdd10")
        assert {item["class_name"] for item in main.detect(None)[0]} == {"traffic light", "car"}

        main.single_model = False
        assert {item["class_name"] for item in main.detect(None)[0]} == {"car"}

        specialist = bridge.RknnConsoleDetector(
            "specialist.rknn", (512, 192), specialist=True,
            class_names=("traffic light",),
        )
        assert specialist.detector is not None
        assert specialist.detect(None)[0]
    finally:
        detector_module.CompetitionDetector = original_detector
