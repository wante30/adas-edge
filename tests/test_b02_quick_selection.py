from __future__ import annotations

import unittest

from tools.b02_quick_selection import aggregate_counts, high_confidence_false_positives


class AggregateCountsTests(unittest.TestCase):
    def test_aggregates_counts_before_calculating_micro_metrics(self) -> None:
        actual = aggregate_counts(
            {
                "car": {"tp": 4, "fp": 2, "fn": 1},
                "person": {"tp": 1, "fp": 3, "fn": 4},
            },
            ("car", "person"),
        )

        self.assertEqual((actual["tp"], actual["fp"], actual["fn"]), (5, 5, 5))
        self.assertAlmostEqual(actual["precision"], 0.5)
        self.assertAlmostEqual(actual["recall"], 0.5)
        self.assertAlmostEqual(actual["f1"], 0.5)

    def test_empty_group_has_zero_metrics(self) -> None:
        actual = aggregate_counts({}, ())
        self.assertEqual((actual["tp"], actual["fp"], actual["fn"]), (0, 0, 0))
        self.assertEqual((actual["precision"], actual["recall"], actual["f1"]), (0.0, 0.0, 0.0))

    def test_false_positive_list_marks_duplicate_and_disjoint_boxes(self) -> None:
        examples = high_confidence_false_positives(
            [
                (
                    "frame.jpg",
                    [
                        (2, 0, 0, 10, 10, 0.9),
                        (2, 0, 0, 10, 10, 0.8),
                        (2, 20, 20, 30, 30, 0.7),
                    ],
                )
            ],
            {2: "car"},
            {
                "images": [{"id": 1, "file_name": "frame.jpg"}],
                "annotations": [
                    {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}
                ],
            },
        )

        self.assertEqual(len(examples), 2)
        self.assertEqual([row["score"] for row in examples], [0.8, 0.7])
        self.assertEqual(examples[0]["max_iou_same_class"], 1.0)
        self.assertEqual(examples[1]["max_iou_same_class"], 0.0)


if __name__ == "__main__":
    unittest.main()
