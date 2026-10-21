from __future__ import annotations

import unittest

import numpy as np

from tools.collect_rknn_zoo_onnx_predictions import decode_model_zoo_outputs


def make_outputs(family: str) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for _ in range(3):
        if family == "yolov6":
            position = np.full((1, 4, 1, 1), 0.1, dtype=np.float32)
        else:
            position = np.full((1, 64, 1, 1), -10.0, dtype=np.float32)
            for side in range(4):
                position[0, side * 16 + 1, 0, 0] = 10.0
        class_scores = np.full((1, 80, 1, 1), 0.01, dtype=np.float32)
        class_scores[0, 2, 0, 0] = 0.9
        score_sum = np.ones((1, 1, 1, 1), dtype=np.float32)
        outputs.extend((position, class_scores, score_sum))
    return outputs


class ModelZooOutputDecodeTests(unittest.TestCase):
    def test_yolov6_optimized_outputs_decode_direct_ltrb_distances(self) -> None:
        detections = decode_model_zoo_outputs(
            make_outputs("yolov6"), "yolov6", input_shape=(640, 640), confidence_threshold=0.25
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0][0], 2)
        self.assertAlmostEqual(detections[0][1], 256.0, places=3)
        self.assertAlmostEqual(detections[0][2], 256.0, places=3)
        self.assertAlmostEqual(detections[0][3], 384.0, places=3)
        self.assertAlmostEqual(detections[0][4], 384.0, places=3)
        self.assertAlmostEqual(detections[0][5], 0.9, places=6)

    def test_yolov8_optimized_outputs_apply_dfl_and_suppress_duplicate_scales(self) -> None:
        detections = decode_model_zoo_outputs(
            make_outputs("yolov8"), "yolov8", input_shape=(640, 640), confidence_threshold=0.25
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0][0], 2)
        self.assertAlmostEqual(detections[0][5], 0.9, places=6)

    def test_model_zoo_decoder_rejects_unexpected_branch_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "9 outputs"):
            decode_model_zoo_outputs([], "yolov6", input_shape=(640, 640))


if __name__ == "__main__":
    unittest.main()
