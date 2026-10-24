from tools.b05c_diagnose import ap50, at_conf, iou


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_ap50_uses_global_confidence_order():
    gt = {
        "a.jpg": [{"bbox": [0, 0, 10, 10], "color": "red"}],
        "b.jpg": [{"bbox": [0, 0, 10, 10], "color": "green"}],
    }
    preds = {
        "a.jpg": [{"bbox": [20, 20, 30, 30], "confidence": 0.9}],
        "b.jpg": [{"bbox": [0, 0, 10, 10], "confidence": 0.8}],
    }
    result = ap50(preds, gt)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert abs(result["ap50"] - 0.2524752475247525) < 1e-6


def test_at_conf_does_not_mix_images():
    gt = {
        "a.jpg": [{"bbox": [0, 0, 10, 10], "color": "red"}],
        "b.jpg": [{"bbox": [0, 0, 10, 10], "color": "green"}],
    }
    preds = {
        "a.jpg": [{"bbox": [0, 0, 10, 10], "confidence": 0.8}],
        "b.jpg": [{"bbox": [20, 20, 30, 30], "confidence": 0.9}],
    }
    result = at_conf(preds, gt, 0.25)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1

