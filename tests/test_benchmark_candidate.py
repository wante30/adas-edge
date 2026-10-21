from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

from deployment.rknn.benchmark_candidate import (
    classwise_nms,
    decode_yolov8_split,
    load_input_tensor,
    parse_args,
    read_npu_frequencies,
    read_npu_load,
    summarize_latency,
)


class CandidateBenchmarkTests(unittest.TestCase):
    def test_npu_run_count_defaults_to_two_hundred(self) -> None:
        args = parse_args(["--model", "candidate.rknn", "--input-npy", "input.npy"])

        self.assertEqual(args.runs, 200)

    def test_pipeline_requires_a_supported_decoder(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--model", "candidate.rknn", "--input-npy", "input.npy", "--image", "frame.jpg"])

    def test_load_input_tensor_accepts_hwc_uint8_as_nhwc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.npy"
            np.save(path, np.zeros((288, 512, 3), dtype=np.uint8))

            tensor = load_input_tensor(path)

        self.assertEqual(tensor.shape, (1, 288, 512, 3))
        self.assertEqual(tensor.dtype, np.uint8)

    def test_load_input_tensor_rejects_non_uint8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.npy"
            np.save(path, np.zeros((1, 288, 512, 3), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "uint8"):
                load_input_tensor(path)

    def test_summarize_latency_reports_tail_percentiles(self) -> None:
        result = summarize_latency([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(result["mean_ms"], 3.0)
        self.assertEqual(result["p50_ms"], 3.0)
        self.assertEqual(result["p95_ms"], 4.8)

    def test_decode_and_classwise_nms_keep_best_per_class(self) -> None:
        boxes = np.array([[[10.0, 10.0, 20.0], [10.0, 10.0, 20.0], [8.0, 8.0, 4.0], [8.0, 8.0, 4.0]]])
        scores = np.array([[[0.90, 0.80, 0.20], [0.10, 0.20, 0.60]]], dtype=np.float32)

        xyxy, confidence, class_ids = decode_yolov8_split([boxes, scores], confidence_threshold=0.25)
        kept = classwise_nms(xyxy, confidence, class_ids, iou_threshold=0.45)

        self.assertEqual(class_ids.tolist(), [0, 0, 1])
        self.assertEqual(kept, [0, 2])

    def test_read_npu_frequencies_finds_devfreq_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frequency = Path(temporary) / "class" / "devfreq" / "fde40000.npu" / "cur_freq"
            frequency.parent.mkdir(parents=True)
            frequency.write_text("800000000\n", encoding="utf-8")

            result = read_npu_frequencies(Path(temporary))

        self.assertEqual(result, {"fde40000.npu": 800_000_000})

    def test_read_npu_load_returns_debugfs_value_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            load = Path(temporary) / "kernel" / "debug" / "rknpu" / "load"
            load.parent.mkdir(parents=True)
            load.write_text("NPU load: 0%, 75%\n", encoding="utf-8")

            result = read_npu_load(Path(temporary))

        self.assertEqual(result, "NPU load: 0%, 75%")


if __name__ == "__main__":
    unittest.main()
