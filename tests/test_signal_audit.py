from tools.evaluate_signal_audit import evaluate


def test_unreviewed_signal_frames_cannot_establish_precision():
    labels = {"ranges": [
        {"start_frame": 0, "end_frame": 1, "vehicle_state": "GREEN"},
        {"start_frame": 2, "end_frame": 3, "vehicle_state": "UNKNOWN"},
    ]}
    result = evaluate({
        0: {"traffic_light_state": "GREEN"},
        1: {"traffic_light_state": "UNKNOWN"},
        2: {"traffic_light_state": "GREEN"},
    }, labels)["vehicle_green"]
    assert (result["tp"], result["fn"], result["fp"]) == (1, 1, 0)
    assert result["recall"] == 0.5
    assert result["precision"] is None
    assert result["f1"] is None
    assert result["excluded_ambiguous_or_unreviewed_frames"] == 2


def test_missing_positive_predictions_and_reviewed_negative_errors_are_counted():
    labels = {"ranges": [
        {"start_frame": 0, "end_frame": 1, "vehicle_state": "GREEN"},
        {"start_frame": 2, "end_frame": 2, "vehicle_state": "RED"},
    ]}
    result = evaluate({
        0: {"traffic_light_state": "GREEN"},
        2: {"traffic_light_state": "GREEN"},
    }, labels)["vehicle_green"]
    assert (result["tp"], result["fn"], result["fp"]) == (1, 1, 1)
    assert result["missing_prediction_frames"] == 1
    assert result["precision"] == result["recall"] == 0.5


def test_audit_can_score_association_decision_field() -> None:
    labels = {"ranges": [{"start_frame": 0, "end_frame": 1, "vehicle_state": "GREEN"}]}
    result = evaluate(
        {0: {"traffic_light_state": "GREEN", "decision": "GREEN"}, 1: {"traffic_light_state": "GREEN", "decision": "UNKNOWN"}},
        labels,
        vehicle_prediction_key="decision",
    )["vehicle_green"]
    assert (result["tp"], result["fn"]) == (1, 1)
