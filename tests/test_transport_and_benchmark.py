from datetime import datetime, timezone
from io import StringIO

import numpy as np

from adas_edge.benchmark import BenchmarkRecorder
from adas_edge.contracts import Frame, PixelFormat
from adas_edge.decision.engine import ControlCommand
from adas_edge.pipeline import PipelineResult
from adas_edge.transport.control import ConsoleControlTransport


def test_console_control_transport_writes_structured_json() -> None:
    stream = StringIO()
    command = ControlCommand("STOP", ("RED_LIGHT",), 0.9, "2026-01-01T00:00:00+00:00")

    ConsoleControlTransport(stream).send(command)

    assert '"action": "STOP"' in stream.getvalue()
    assert '"RED_LIGHT"' in stream.getvalue()


def test_benchmark_recorder_preserves_camera_and_latency_schema(tmp_path) -> None:
    frame = Frame("rear", 3, datetime.now(timezone.utc), np.zeros((4, 4, 3), np.uint8), PixelFormat.BGR888)
    command = ControlCommand("FORWARD", (), 1.0, "2026-01-01T00:00:00+00:00")
    result = PipelineResult(frame, (), command, {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0, "end_to_end": 6.0}, "test")
    recorder = BenchmarkRecorder("test", 640, "FP32")

    row = recorder.row(result, queue_latency_ms=0.5, dropped_frames=2)
    BenchmarkRecorder.append_csv(tmp_path / "bench.csv", row)

    assert row["camera_id"] == "rear"
    assert row["dropped_frames"] == 2
    assert "queue_latency_ms" in (tmp_path / "bench.csv").read_text(encoding="utf-8")
