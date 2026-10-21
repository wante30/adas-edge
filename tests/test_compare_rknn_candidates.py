from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr

from deployment.rknn.compare_rknn_candidates import (
    BDD_NAMES,
    BDD_SOURCE_NAMES,
    COCO_FOCUS_NAMES,
    parse_args,
    summarize_latency,
)


class CompareRknnCandidatesTests(unittest.TestCase):
    def test_class_mappings_keep_rider_separate_and_use_model_specific_ids(self) -> None:
        self.assertEqual(BDD_NAMES[1], "rider")
        self.assertEqual(BDD_NAMES[8], "traffic light")
        self.assertEqual(BDD_SOURCE_NAMES[6], "motor")
        self.assertEqual(BDD_NAMES[6], "motorcycle")
        self.assertEqual(BDD_SOURCE_NAMES[7], "bike")
        self.assertEqual(BDD_NAMES[7], "bicycle")
        self.assertEqual(COCO_FOCUS_NAMES[9], "traffic light")
        self.assertNotIn(4, COCO_FOCUS_NAMES)

    def test_latency_summary_reports_mean_and_tail(self) -> None:
        result = summarize_latency([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result["mean_ms"], 3.0)
        self.assertEqual(result["p95_ms"], 4.8)

    def test_parse_args_rejects_zero_runs(self) -> None:
        args = [
            "--baseline-model", "base.rknn", "--candidate-model", "candidate.rknn",
            "--manifest", "manifest.json", "--inputs-dir", "inputs", "--images-dir", "images",
            "--runs", "0", "--output", "result.json",
        ]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(args)


if __name__ == "__main__":
    unittest.main()
