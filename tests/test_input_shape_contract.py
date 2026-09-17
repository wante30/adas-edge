"""The input-shape contract: square stays compatible, rectangular must be explicit.

Both copies of the contract (``adas_edge.input_shape`` for the standalone PC
library, ``competition_ai.input_shape`` for the competition and board path) are
checked against the same table, so the deliberate duplication cannot drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]

# 640x360 is NOT here on purpose: 360 is not a multiple of 32, and Ultralytics
# rounds it to 640x384 behind the caller's back. 640x352 is the nearest legal
# height, and it is the high-quality FRONT candidate this project uses.
RECTANGULAR_CANVASES = [(320, 192), (416, 256), (512, 288), (640, 352)]
SQUARE_CANVASES = [(320, 320), (416, 416), (512, 512), (640, 640)]


def _load_competition_module():
    path = REPO / "competition_ai" / "input_shape.py"
    spec = importlib.util.spec_from_file_location("competition_input_shape", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_library_module():
    path = REPO / "src" / "adas_edge" / "input_shape.py"
    spec = importlib.util.spec_from_file_location("adas_edge_input_shape", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["library", "competition"])
def contract(request):
    return _load_library_module() if request.param == "library" else _load_competition_module()


def test_both_copies_agree_on_the_shared_table() -> None:
    library = _load_library_module()
    competition = _load_competition_module()
    accepted = [320, "320", (512, 288), [512, 288], "512x288", "512X288", "512×288", "512*288", "512,288"]
    rejected = [0, -32, 300, "512x300", "0x288", "-16x288", "512x288x3", "abc", (512,), (512, 288, 3)]
    for value in accepted:
        assert library.validate_model_shape(value) == competition.validate_model_shape(value), value
        assert library.to_ultralytics_imgsz(value) == competition.to_ultralytics_imgsz(value), value
        assert library.format_shape(value) == competition.format_shape(value), value
    for value in rejected:
        for module in (library, competition):
            with pytest.raises(ValueError):
                module.validate_model_shape(value)


def test_square_configuration_keeps_its_historical_meaning(contract) -> None:
    assert contract.normalize_shape(320) == (320, 320)
    assert contract.validate_model_shape(320) == (320, 320)
    assert contract.validate_model_shape("320") == (320, 320)
    # square shapes stay a bare int on the Ultralytics side, as they always did
    assert contract.to_ultralytics_imgsz(320) == 320
    assert contract.format_shape(320) == "320"


def test_rectangular_canvases_are_width_height_in_that_order(contract) -> None:
    for width, height in RECTANGULAR_CANVASES:
        assert contract.validate_model_shape((width, height)) == (width, height)
        assert contract.validate_model_shape([width, height]) == (width, height)
        assert contract.validate_model_shape(f"{width}x{height}") == (width, height)


def test_ultralytics_imgsz_flips_to_height_width_for_rectangles(contract) -> None:
    for width, height in RECTANGULAR_CANVASES:
        # Ultralytics takes (height, width); getting this backwards would silently
        # transpose every rectangular canvas.
        assert contract.to_ultralytics_imgsz((width, height)) == (height, width)


def test_format_shape_renders_width_first(contract) -> None:
    for width, height in RECTANGULAR_CANVASES:
        assert contract.format_shape((width, height)) == f"{width}x{height}"


def test_square_canvases_pass_through_unchanged(contract) -> None:
    for width, height in SQUARE_CANVASES:
        assert contract.validate_model_shape((width, height)) == (width, height)
        assert contract.format_shape((width, height)) == str(width)


def test_normalize_is_pure_and_validate_is_policy(contract) -> None:
    # pure: any positive shape, so unit-scale tensors stay decodable
    assert contract.normalize_shape(8) == (8, 8)
    # policy: the graph shape must satisfy the stride
    with pytest.raises(ValueError):
        contract.validate_model_shape(8)


def test_rejects_non_integer_and_non_positive_sides(contract) -> None:
    for value in [(0, 288), (512, 0), (-512, 288), (512, -288)]:
        with pytest.raises(ValueError):
            contract.normalize_shape(value)
    with pytest.raises(ValueError):
        contract.normalize_shape(True)


def test_parse_shape_text_handles_the_deployment_spellings(contract) -> None:
    assert contract.parse_shape_text("512x288") == (512, 288)
    assert contract.parse_shape_text("512X288") == (512, 288)
    assert contract.parse_shape_text("512×288") == (512, 288)
    assert contract.parse_shape_text("512*288") == (512, 288)
    assert contract.parse_shape_text("320") == 320
    for bad in ("", "abc", "512x", "x288", "512x288x3"):
        with pytest.raises(ValueError):
            contract.parse_shape_text(bad)


def test_640x360_is_not_a_legal_static_canvas() -> None:
    """360 is not divisible by 32, so 640x360 cannot exist as a frozen graph.

    Kept as an explicit test because it is the kind of shape an experiment plan
    writes down naturally, and ``check_imgsz`` would silently turn it into 640x384
    instead of failing.
    """
    library = _load_library_module()
    competition = _load_competition_module()

    for module in (library, competition):
        with pytest.raises(ValueError):
            module.validate_model_shape((640, 360))
        # the nearest legal neighbours on either side of 360
        assert module.validate_model_shape((640, 352)) == (640, 352)
        assert module.validate_model_shape((640, 384)) == (640, 384)


def test_only_512x288_is_an_exact_16_by_9_stride_legal_canvas() -> None:
    """For w/32 and h/32 both integral, 16:9 forces w=512k, h=288k."""
    library = _load_library_module()
    legal_16_9 = []
    for scale in range(1, 21):
        width, height = 512 * scale, 288 * scale
        try:
            library.validate_model_shape((width, height))
        except ValueError:
            continue
        legal_16_9.append((width, height))
    assert legal_16_9[0] == (512, 288)
    # the next one is 1024x576, which is far beyond an RK3566 single-stream budget
    assert legal_16_9[1] == (1024, 576)
