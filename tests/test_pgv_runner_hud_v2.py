from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import run_competition_console as runner
from tools.competition_console.controls import Controller
from tools.competition_console.navigation_reader import NavigationObservation
from tools.competition_console.pipeline import FrontSnapshot
from tools.competition_console.hud_diagnostics import DiagnosticStatus
from competition_ai.bound_signal import SignalObservation
from competition_ai.hazard_events import HazardObservation
from competition_ai.scenario import Light


class FakePgvCapture:
    def __init__(self, road, navigation, frame_id=42, timestamp_ns=8_000_000_000):
        self.road = road
        self.navigation_image = navigation
        self.frame_id = frame_id
        self.timestamp_ns = timestamp_ns
        self.age_ms = 3.0
        self.dropped_frames = 0
        self.error = None

    def read(self):
        return True, self.road

    def release(self):
        pass


def test_parser_exposes_fixed_q3_road_q1_navigation_hud_transport(tmp_path):
    args = runner.parser().parse_args([
        "--pgv", "capture.pgv", "--pgv-quadrant", "3",
        "--pgv-navigation-quadrant", "1", "--fpga-hud",
        "--bar1-resource", str(tmp_path / "resource1"),
    ])

    assert args.pgv_quadrant == 3
    assert args.pgv_navigation_quadrant == 1
    assert args.fpga_hud is True
    assert args.bar1_resource == tmp_path / "resource1"
    assert args.mode is None


def test_front_snapshot_ages_and_expires_detections_signal_and_hazards():
    sample_ns = 2_000_000_000
    signal = SignalObservation(light=Light.GREEN, raw_light=Light.GREEN,
                               timestamp_ns=sample_ns, associated=True, reason="observed")
    hazards = HazardObservation(timestamp_ns=sample_ns,
                                events=({"kind": "LEAD_RAPID_CLOSING"},), reason="tracking")
    snapshot = FrontSnapshot(
        main=[{"class_name": "car"}], specialist=[{"class_name": "traffic light"}],
        signal=signal, main_age_ms=0.0, specialist_age_ms=0.0,
        binding_source="UNBOUND", hazards=hazards, sample_timestamp_ns=sample_ns,
    )

    fresh = snapshot.at(sample_ns + 200_000_000)
    assert fresh.main_age_ms == pytest.approx(200.0)
    assert fresh.specialist_age_ms == pytest.approx(200.0)
    assert fresh.signal.light is Light.GREEN
    assert len(fresh.hazards.events) == 1

    stale = snapshot.at(sample_ns + 401_000_000)
    assert stale.main
    assert stale.specialist == []
    assert stale.signal.light is Light.UNKNOWN
    assert not stale.signal.associated
    assert stale.signal.reason == "observation_expired"
    assert stale.hazards.events == ()
    assert stale.hazards.reason == "observation_expired"

    expired = snapshot.at(sample_ns + 751_000_000)
    assert expired.main == []


def test_hardware_live_keyboard_blocks_controls_but_allows_signal_binding():
    controller = Controller()
    original_controls = controller.controls

    for key in ("1", "2", "3", "s", "g", "a", "q", "e", "j", "l", "t"):
        runner.dispatch_key(controller, key, hardware_live=True)

    assert controller.controls == original_controls
    assert controller.binding is False

    runner.dispatch_key(controller, ord("b"), hardware_live=True)

    assert controller.controls == original_controls
    assert controller.binding is True


def test_live_signal_roi_binding_uses_pgv_timestamp_not_zero_fps():
    class Signal:
        def __init__(self):
            self.update_args = None

        def update(self, frame, frame_id, timestamp_ns):
            self.update_args = (frame, frame_id, timestamp_ns)

    class Pipeline:
        def __init__(self):
            self.signal = Signal()
            self.bind_args = None
            self.bindings = {}

        def bind(self, frame, bbox, lanes):
            self.bind_args = (frame, bbox, lanes)
            self.bindings[lanes[0]] = (self.signal, "USER_SELECTED_HORIZONTAL_RYG_HEAD")

    frame = np.zeros((100, 160, 3), np.uint8)
    pipeline = Pipeline()

    runner.bind_signal_roi(pipeline, frame, [10, 10, 70, 30], "left", 42, 8_000_000_000)

    assert pipeline.bind_args == (frame, [10, 10, 70, 30], ["left"])
    assert pipeline.signal.update_args == (frame, 42, 8_000_000_000)


def test_live_pgv_processes_q3_and_q1_and_sends_only_perception(tmp_path, monkeypatch):
    road = np.zeros((360, 640, 3), np.uint8)
    road[14:24, 14:26] = (0, 0, 255)
    navigation = np.full((360, 640, 3), 127, np.uint8)
    capture = FakePgvCapture(road, navigation)
    calls = {"detector_shapes": [], "navigation": [], "hud": []}

    class FakeDetector:
        def __init__(self, model_path, input_size, specialist=False):
            self.specialist = specialist
            self.session = None

        def detect(self, frame):
            calls["detector_shapes"].append(frame.shape[:2])
            if self.specialist:
                return ([{"class_name": "traffic light", "confidence": 0.9,
                          "bbox": [10, 10, 30, 40]}], 1.0)
            return ([{"class_name": "person", "confidence": 0.8,
                      "bbox": [100, 100, 130, 200]}], 1.0)

    class FakeNavigationReader:
        def update(self, image, frame_id, timestamp_ns):
            calls["navigation"].append((image.shape, frame_id, timestamp_ns))
            return NavigationObservation(
                direction="none", distance_m=None, road_name="", valid=False,
                confidence=0.0, reason="unconfirmed", source_frame_id=frame_id,
                timestamp_ns=timestamp_ns, raw_text=[], inference_ms=0.1,
            )

        def reset(self):
            pass

    class FakeHudSender:
        def __init__(self, bus):
            self.pending = None
            self.closed = False

        def probe(self):
            pass

        def commit(self, ext_request, **kwargs):
            calls["hud"].append((ext_request, kwargs))
            return SimpleNamespace(
                phase="COMMITTED", sequence=1, ext_request=ext_request,
                time0=0, time1=0, committed=True, reason="",
                applied_seq=1, mailbox_status=2,
            )

        def close(self):
            self.closed = True

    main_model = tmp_path / "main.rknn"
    specialist_model = tmp_path / "specialist.rknn"
    main_model.write_bytes(b"main")
    specialist_model.write_bytes(b"specialist")
    monkeypatch.setattr(runner, "pgv_capture", lambda *args: (capture, 0.0, None))
    monkeypatch.setattr(runner, "Bar1MmapBus", lambda path: object())
    monkeypatch.setattr(runner, "HudV2Sender", FakeHudSender)
    monkeypatch.setattr(runner, "PcDetector", FakeDetector)
    monkeypatch.setattr(runner, "NavigationReader", FakeNavigationReader, raising=False)
    monkeypatch.setattr(runner, "apply_navigation", lambda controls, observation: controls)

    args = runner.parser().parse_args([
        "--pgv", "latest.pgv", "--pgv-quadrant", "3",
        "--pgv-navigation-quadrant", "1", "--fpga-hud",
        "--bar1-resource", "unused", "--detector-backend", "onnx",
        "--main-model", str(main_model), "--specialist-model", str(specialist_model),
        "--headless", "--max-frames", "1", "--output", str(tmp_path / "run"),
    ])

    report = runner.run(args)

    assert calls["navigation"] == [((540, 960, 3), 42, 8_000_000_000)]
    assert report["processed_frames"] == 1
    assert report["pcie_source"]["road_quadrant"] == 3
    assert report["pcie_source"]["navigation_quadrant"] == 1
    assert report["fpga_hud"]["applied_transactions"] == 1
    assert calls["hud"][0][1]["time_enable"] is False
    assert calls["hud"][0][0] == 0xB
    assert calls["hud"][0][0] & (1 << 11) == 0
    assert calls["detector_shapes"] == [(360, 640), (216, 640)]

    rows = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text().splitlines()]
    assert rows[0]["source_timestamp_ns"] == 8_000_000_000
    assert rows[0]["navigation"]["source_frame_id"] == 42
    assert rows[0]["fpga_hud_receipt"]["phase"] == "COMMITTED"


@pytest.mark.parametrize("args", [
    ["--pgv", "latest.pgv", "--fpga-hud", "--bar1-resource", "/dev/resource1", "--profile", "scene.json"],
    ["--pgv", "latest.pgv", "--fpga-hud", "--bar1-resource", "/dev/resource1", "--mock-hud"],
    ["--video", "clip.mp4", "--fpga-hud", "--bar1-resource", "/dev/resource1"],
])
def test_hardware_hud_rejects_non_live_or_simulated_inputs(args):
    parsed = runner.parser().parse_args(args)

    with pytest.raises(ValueError, match="FPGA HUD requires live PGV perception"):
        runner.validate_hardware_hud_args(parsed)
