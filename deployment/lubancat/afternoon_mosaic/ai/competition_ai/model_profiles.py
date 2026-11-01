"""Explicit model class profiles.

The BDD RKNN model must not be identified by a filename substring.  Callers
select a profile by name and the profile is validated before inference.
"""
from __future__ import annotations

from typing import Iterable

COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

BDD10 = (
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
)

PROFILES = {
    "coco80": COCO80,
    "bdd10": BDD10,
}


def get_profile(name: str) -> tuple[str, ...]:
    try:
        names = PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown model profile: {name!r}") from exc
    if not names or len(set(names)) != len(names):
        raise ValueError(f"invalid class profile: {name!r}")
    return names


def validate_classes(names: Iterable[str], expected: int) -> tuple[str, ...]:
    result = tuple(str(value) for value in names)
    if len(result) != expected:
        raise ValueError(f"class count {len(result)} != expected {expected}")
    if len(set(result)) != len(result):
        raise ValueError("duplicate class names")
    return result
