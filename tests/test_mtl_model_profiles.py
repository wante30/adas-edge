"""Model registry checks for the 2026-10-31 C API / MTL integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_profiles_module():
    path = Path(__file__).resolve().parents[1] / "deployment/lubancat/qt_gpu_probe/model_profiles.py"
    spec = importlib.util.spec_from_file_location("adas_model_profiles", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_model_is_b05n_p2():
    profiles = _load_profiles_module()
    assert profiles.DEFAULT_MODEL_ID == "B"
    assert profiles.B05N_P2_FILENAME.endswith(".rknn")


def test_mtl_and_capi_profiles_are_registered():
    profiles = _load_profiles_module()
    registry = {
        profile.id: profile
        for profile in profiles.build_model_profiles(
            "/home/cat/cityscapes_arch_compare_20261027/models", "/home/cat"
        )
    }
    assert len(registry) == 21
    assert registry["B"].backend == "rknn_lite"
    assert registry["B"].task == "detect"
    assert registry["B-capi"].backend == "rknn_capi"
    assert registry["B-capi"].task == "detect"
    for model_id in ("MTL-FULL", "MTL-FAST"):
        assert registry[model_id].backend == "rknn_capi"
        assert registry[model_id].task == "mtl"
        assert registry[model_id].specialist_model is None


def test_wire_options_fit_the_bounded_status_packet():
    profiles = _load_profiles_module()
    options = [
        profile.wire()
        for profile in profiles.build_model_profiles(
            "/home/cat/cityscapes_arch_compare_20261027/models", "/home/cat"
        )
    ]
    assert all(set(item) == {
        "id", "label", "available", "legacy", "has_specialist", "reason"
    } for item in options)
