from pathlib import Path
import sys


QT_PROBE_DIR = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "lubancat"
    / "qt_gpu_probe"
)
sys.path.insert(0, str(QT_PROBE_DIR))
try:
    import model_profiles
finally:
    sys.path.pop(0)


def _touch(root: Path, relative_paths):
    for relative in relative_paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")


def test_registry_exposes_four_formal_schemes_and_installed_legacy_models(tmp_path):
    current = tmp_path / "models"
    legacy = tmp_path / "cat"
    _touch(current, [
        "bdd_yolov8n_512x288_raw_heads_int8.rknn",
        "b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn",
        "b05l_v8n_p2_raw_heads_int8_500_20261026.rknn",
    ])
    _touch(legacy, [
        "adas-edge-rect/rect_models/bdd_yolov8n_512x288_int8_kl.rknn",
        "b03e_quant_20261022/bdd_yolov8n_512x288_int8.rknn",
        "adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_320x192.rknn",
        "adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_416x256.rknn",
        "adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_512x288.rknn",
        "adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_512x288_mmse.rknn",
        "adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_640x352.rknn",
        "adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn",
        "adas-edge-rect/tl_check/tl_specialist_512x192_logits.rknn",
        "adas-edge-rect/tl_check/tl_specialist_512x192_hybrid.rknn",
        "adas-edge-rect/tl_check/tl_specialist_rk3566_int8_split_512x192.rknn",
        "adas-edge-rect/tl_check/tl_specialist_512x192_traincalib.rknn",
    ])

    profiles = {profile.id: profile for profile in model_profiles.build_model_profiles(current, legacy)}

    assert {"A", "B", "C", "D"}.issubset(profiles)
    assert profiles["A"].available
    assert profiles["B"].available
    assert profiles["C"].available
    assert profiles["D"].available
    assert profiles["A"].main_model == profiles["C"].main_model
    assert profiles["B"].main_model == profiles["D"].main_model
    assert profiles["A"].specialist_model is None
    assert profiles["B"].specialist_model is None
    assert profiles["C"].specialist_model is not None
    assert profiles["D"].specialist_model is not None
    assert profiles["B"].class_profile == "bdd10"
    assert profiles["legacy-bdd-kl"].legacy
    assert profiles["legacy-bdd-int8"].legacy
    assert profiles["legacy-yolo11n"].legacy
    assert profiles["legacy-yolo11n-specialist"].legacy
    assert profiles["legacy-yolo11n-320"].input_size == (320, 192)
    assert profiles["legacy-yolo11n-416"].input_size == (416, 256)
    assert profiles["legacy-yolo11n-mmse"].available
    assert profiles["legacy-yolo11n-640"].input_size == (640, 352)
    assert profiles["legacy-bdd-specialist-int8-split"].available


def test_missing_model_files_remain_visible_but_are_not_selectable(tmp_path):
    profiles = {profile.id: profile for profile in model_profiles.build_model_profiles(
        tmp_path / "missing-models", tmp_path / "missing-legacy"
    )}

    assert {"A", "B", "C", "D"}.issubset(profiles)
    assert not profiles["A"].available
    assert not profiles["D"].available


def test_model_switch_request_round_trips_and_rejects_malformed_packets():
    packet = model_profiles.encode_model_switch_request(0x01020304, "legacy-yolo11n")

    assert packet[:4] == b"MVM1"
    assert model_profiles.decode_model_switch_request(packet) == {
        "request_id": 0x01020304,
        "profile_id": "legacy-yolo11n",
    }
    assert model_profiles.decode_model_switch_request(packet[:-1]) is None
    assert model_profiles.decode_model_switch_request(b"BAD!" + packet[4:]) is None
    assert model_profiles.encode_model_switch_request(3, "../invalid") is None


def test_model_switch_reply_round_trips_state_and_active_profile():
    packet = model_profiles.encode_model_switch_reply(
        27,
        state="APPLIED",
        active_id="D",
        requested_id="D",
        message="",
    )

    assert packet[:4] == b"MMS1"
    assert model_profiles.decode_model_switch_reply(packet) == {
        "request_id": 27,
        "state": "APPLIED",
        "active_id": "D",
        "requested_id": "D",
        "message": "",
    }
    assert model_profiles.decode_model_switch_reply(packet[:-1]) is None
