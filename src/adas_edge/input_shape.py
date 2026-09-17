"""Input-shape contract for the standalone ``adas_edge`` library.

This mirrors ``competition_ai.input_shape`` on purpose. The two packages must not
import each other: ``adas_edge`` is the installed PC baseline and has to work from
any working directory, while ``competition_ai`` and the board scripts are loaded
relative to the repository root. Duplicating the contract keeps both importable,
and ``tests/test_input_shape_contract.py`` asserts the two copies agree on a shared
table of inputs, so drift becomes a test failure instead of a silent difference in
what "320" means.

``320`` still means 320x320. A rectangular input has to be written explicitly as
``(width, height)``.
"""
from __future__ import annotations

MIN_SIDE = 32
STRIDE = 32


def normalize_shape(size) -> tuple[int, int]:
    """Return ``(width, height)`` for an int or a two-item sequence."""
    if isinstance(size, bool):
        raise ValueError(f"input shape must be int or (width, height), got {size!r}")
    if isinstance(size, int):
        width = height = size
    else:
        try:
            values = list(size)
        except TypeError as exc:
            raise ValueError(f"input shape must be int or (width, height), got {size!r}") from exc
        if len(values) != 2:
            raise ValueError(f"input shape must be int or (width, height), got {size!r}")
        try:
            width, height = int(values[0]), int(values[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"input shape must contain integers, got {size!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"input shape must be positive, got {(width, height)}")
    return width, height


def parse_shape_text(text: str) -> int | tuple[int, int]:
    """Parse ``"320"`` or ``"512x288"`` (also ``X``/``×``/``*``/``,``) into a shape value."""
    clean = text.strip().lower().replace("×", "x").replace("*", "x").replace(",", "x")
    if "x" in clean:
        parts = [part.strip() for part in clean.split("x")]
        if len(parts) == 2 and all(part.lstrip("+-").isdigit() for part in parts):
            return (int(parts[0]), int(parts[1]))
    if clean.lstrip("+-").isdigit():
        return int(clean)
    raise ValueError(f"cannot parse model shape {text!r}; expected e.g. 512x288 or 320")


def validate_model_shape(shape) -> tuple[int, int]:
    """Return ``(width, height)`` after enforcing the >=32 / multiple-of-32 rule."""
    resolved = parse_shape_text(shape) if isinstance(shape, str) else shape
    width, height = normalize_shape(resolved)
    if width < MIN_SIDE or height < MIN_SIDE or width % STRIDE or height % STRIDE:
        raise ValueError(
            f"model input must be >={MIN_SIDE} and divisible by {STRIDE}, got {(width, height)}"
        )
    return width, height


def to_ultralytics_imgsz(shape) -> int | tuple[int, int]:
    """Convert to the ``imgsz`` Ultralytics expects: an int, or ``(height, width)``."""
    width, height = validate_model_shape(shape)
    return height if width == height else (height, width)


def format_shape(shape) -> str:
    """Render ``(width, height)`` as ``WxH``; a square value stays a single number."""
    width, height = validate_model_shape(shape)
    return str(width) if width == height else f"{width}x{height}"
