import numpy as np

from competition_ai.autoware_classifier import ConservativeFusionBackend, _probabilities, parse_label
from competition_ai.multi_signal import Attributes
from tools.offline_signal_models import OfflineDetector


def test_official_single_labels_keep_color_and_shape() -> None:
    assert parse_label("red") == ("RED", "ROUND", [], "AUTOWARE_COLOR_ONLY")
    assert parse_label("left,red") == ("RED", "LEFT_ARROW", ["LEFT_ARROW"], "AUTOWARE_SINGLE_DIRECTION")


def test_official_compound_label_is_not_forced_to_straight() -> None:
    color, shape, directions, reason = parse_label("left,red,straight")
    assert color == "RED"
    assert shape == "UNKNOWN"
    assert directions == ["LEFT_ARROW", "STRAIGHT_ARROW"]
    assert reason == "AUTOWARE_COMBINATION"


def test_autoware_probability_output_is_not_softmaxed_twice() -> None:
    probabilities, semantics = _probabilities(np.array([0.8, 0.1, 0.1], dtype=np.float32))
    assert semantics == "probabilities"
    assert np.argmax(probabilities) == 0
    assert float(probabilities[0]) > 0.79


def test_fusion_does_not_promote_color_only_round_on_small_roi() -> None:
    class Legacy:
        def classify(self, roi, *, request=None):
            return Attributes(color="RED", shape="UNKNOWN", reason="INSUFFICIENT_SHAPE_PIXELS")

    class Learned:
        def classify(self, roi, *, request=None):
            return Attributes(color="RED", shape="ROUND", reason="AUTOWARE_COLOR_ONLY")

    fused = ConservativeFusionBackend(Legacy(), Learned()).classify(np.zeros((4, 4, 3), np.uint8))
    assert fused.color == "RED"
    assert fused.shape == "UNKNOWN"


def test_fusion_keeps_legacy_arrow_against_color_only_cnn() -> None:
    class Legacy:
        def classify(self, roi, *, request=None):
            return Attributes(color="RED", shape="LEFT_ARROW", shape_confidence=.9,
                              reason="GLOW_CONTOUR_TEMPLATE")

    class Learned:
        def classify(self, roi, *, request=None):
            return Attributes(color="RED", shape="ROUND", reason="AUTOWARE_COLOR_ONLY")

    fused = ConservativeFusionBackend(Legacy(), Learned()).classify(np.zeros((8, 8, 3), np.uint8))
    assert fused.color == "RED"
    assert fused.shape == "LEFT_ARROW"


def test_diagnostic_nms_trace_preserves_suppressed_candidate() -> None:
    boxes = np.array([[[50., 50.], [70., 70.], [20., 20.], [20., 20.]]], dtype=np.float32)
    scores = np.array([[[.9, .8], [.1, .1]]], dtype=np.float32)
    pre, post, suppressed = OfflineDetector._trace_rows(boxes, scores, score_floor=.001,
                                                         nms_iou=.45, score_logits=False)
    assert len(pre) == 2
    assert len(post) == 1
    assert len(suppressed) == 1
