import numpy as np

from tools.b05k_common import (
    decode_yolo11,
    evaluate_predictions,
    prepare_input,
    should_run_inference,
)


def test_letterbox_and_resize_map_network_boxes_to_original_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)

    letterbox_tensor, letterbox = prepare_input(frame, preprocess="letterbox")
    resize_tensor, resize = prepare_input(frame, preprocess="resize")

    assert letterbox_tensor.shape == resize_tensor.shape == (1, 3, 192, 512)
    assert np.isclose(letterbox_tensor[0, 0, 30, 30], 30 / 255)
    assert letterbox.network_to_frame([256, 9, 512, 182]) == [960.0, 0.0, 1920.0, 648.75]
    assert resize.network_to_frame([256, 96, 512, 192]) == [960.0, 324.0, 1920.0, 648.0]


def test_yolo11_scores_are_used_as_probabilities_without_sigmoid():
    raw = np.array([[[256.0], [96.0], [40.0], [24.0], [0.2]]], dtype=np.float32)

    assert np.isclose(decode_yolo11(raw, conf=0.15, iou=0.7)[0]["confidence"], 0.2)
    assert decode_yolo11(raw, conf=0.25, iou=0.7) == []


def test_one_prediction_cannot_match_two_ground_truth_boxes():
    ground_truth = {
        "image": [
            {"bbox": [0.0, 0.0, 10.0, 10.0]},
            {"bbox": [1.0, 0.0, 11.0, 10.0]},
        ]
    }
    predictions = {
        "image": [{"bbox": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9}]
    }

    metrics = evaluate_predictions(ground_truth, predictions, conf=0.25, iou=0.5)

    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 0, 1)
    assert sum(group["tp"] for group in metrics["size_groups"].values()) == metrics["tp"]


def test_ap50_uses_global_confidence_order_across_images():
    ground_truth = {"positive": [{"bbox": [0.0, 0.0, 10.0, 10.0]}], "negative": []}
    predictions = {
        "positive": [{"bbox": [0.0, 0.0, 10.0, 10.0], "confidence": 0.8}],
        "negative": [{"bbox": [40.0, 40.0, 50.0, 50.0], "confidence": 0.9}],
    }

    metrics = evaluate_predictions(ground_truth, predictions, conf=0.25, iou=0.5)

    assert np.isclose(metrics["ap50"], 0.5)


def test_one_hz_schedule_runs_once_per_source_second():
    assert should_run_inference(0, 30.0, "1hz")
    assert not should_run_inference(29, 30.0, "1hz")
    assert should_run_inference(30, 30.0, "1hz")
    assert should_run_inference(60, 30.0, "1hz")
    assert should_run_inference(29, 30.0, "every-frame")
