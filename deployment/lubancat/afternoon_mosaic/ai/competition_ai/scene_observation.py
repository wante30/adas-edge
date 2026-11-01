"""Platform-neutral road scene observation.

Both platform adapters end here:

    Windows: Qt frame -> ONNX/DirectML -> decoded rows + [3,H,W] masks
    RK3566:  PCIe frame -> RKNN C API -> decoded rows + [3,H,W] masks

Scene algorithms (RoadScene, hazard events, Tesla geometry) must consume this
contract only. They must not import ONNX Runtime, RKNN, Qt, FFmpeg or BAR1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(value), 2) for value in self.bbox],
        }
        if self.track_id is not None:
            item["track_id"] = int(self.track_id)
        return item


@dataclass
class SceneObservation:
    source_id: str
    generation: int
    frame_id: int
    media_s: float
    detections: list[Detection] = field(default_factory=list)
    lane_cat: np.ndarray | None = None
    lane_style: np.ndarray | None = None
    drivable: np.ndarray | None = None
    letterbox: dict[str, float] = field(default_factory=dict)
    source_width: int = 1280
    source_height: int = 720
    traffic_light_state: str = "UNKNOWN"
    source_valid: bool = True

    @property
    def masks_available(self) -> bool:
        return all(mask is not None for mask in
                   (self.lane_cat, self.lane_style, self.drivable))

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-safe handoff for the Qt packet and scene algorithms."""
        def shape(mask: np.ndarray | None) -> list[int]:
            return [] if mask is None else [int(value) for value in mask.shape]

        return {
            "source_id": self.source_id,
            "generation": int(self.generation),
            "frame_id": int(self.frame_id),
            "media_s": round(float(self.media_s), 3),
            "detections": [item.as_dict() for item in self.detections],
            "lane_cat_shape": shape(self.lane_cat),
            "lane_style_shape": shape(self.lane_style),
            "drivable_shape": shape(self.drivable),
            "letterbox": {key: float(value)
                          for key, value in self.letterbox.items()},
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "traffic_light_state": self.traffic_light_state,
            "source_valid": bool(self.source_valid),
        }


def _detections_from_rows(rows: Iterable[dict]) -> list[Detection]:
    result: list[Detection] = []
    for row in rows:
        box = row.get("bbox")
        if box is None:
            x, y = float(row["x"]), float(row["y"])
            box = (x, y, x + float(row["width"]), y + float(row["height"]))
        track = row.get("trackId", row.get("track_id"))
        result.append(Detection(
            class_name=str(row.get("className", row.get("class_name", ""))),
            confidence=float(row.get("confidence", 0.0)),
            bbox=tuple(float(value) for value in box),
            track_id=int(track) if isinstance(track, (int, float)) else None,
        ))
    return result


def _split_masks(masks: np.ndarray | None):
    if masks is None:
        return None, None, None
    array = np.asarray(masks)
    if array.ndim != 3 or array.shape[0] < 3:
        raise ValueError(f"masks must be [3,H,W], got {array.shape}")
    return array[0], array[1], array[2]


def from_windows_mtl(source_id: str, generation: int, frame_id: int,
                     media_s: float, rows: Sequence[dict],
                     masks: np.ndarray | None, meta: dict | None,
                     scale: float, pad_x: int, pad_y: int,
                     source_width: int = 1280, source_height: int = 720,
                     traffic_light_state: str = "UNKNOWN",
                     source_valid: bool = True) -> SceneObservation:
    """Windows ONNX adapter. Rows are in source-crop pixels after decode."""
    lane, style, drivable = _split_masks(masks)
    letterbox = {
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "mask_width": float(meta["mask_width"]) if meta else 0.0,
        "mask_height": float(meta["mask_height"]) if meta else 0.0,
    }
    return SceneObservation(
        source_id=source_id, generation=int(generation), frame_id=int(frame_id),
        media_s=float(media_s), detections=_detections_from_rows(rows),
        lane_cat=lane, lane_style=style, drivable=drivable,
        letterbox=letterbox, source_width=int(source_width),
        source_height=int(source_height),
        traffic_light_state=traffic_light_state,
        source_valid=source_valid)


def from_rknn_capi(source_id: str, generation: int, frame_id: int,
                   media_s: float, rows: Sequence[dict],
                   lane_cat: np.ndarray | None,
                   lane_style: np.ndarray | None,
                   drivable: np.ndarray | None,
                   letterbox: dict[str, float],
                   source_width: int = 1280, source_height: int = 720,
                   traffic_light_state: str = "UNKNOWN",
                   source_valid: bool = True) -> SceneObservation:
    """RK3566 RKNN C API adapter.

    ``rows`` and masks are already in source-frame coordinates; the C API path
    must not reverse-parse a colorized overlay PNG.
    """
    return SceneObservation(
        source_id=source_id, generation=int(generation), frame_id=int(frame_id),
        media_s=float(media_s), detections=_detections_from_rows(rows),
        lane_cat=lane_cat, lane_style=lane_style, drivable=drivable,
        letterbox={key: float(value) for key, value in letterbox.items()},
        source_width=int(source_width), source_height=int(source_height),
        traffic_light_state=traffic_light_state,
        source_valid=bool(source_valid))
