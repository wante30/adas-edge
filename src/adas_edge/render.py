from __future__ import annotations

import cv2
import numpy as np

from adas_edge.pipeline import PipelineResult


def draw_result(result: PipelineResult) -> np.ndarray:
    canvas = result.frame.as_bgr().copy()
    for detection in result.detections:
        x1, y1, x2, y2 = (int(value) for value in detection.bbox_xyxy)
        color = (0, 0, 255) if detection.class_name == "person" else (0, 200, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{detection.class_name} {detection.confidence:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    cv2.putText(
        canvas,
        f"{result.command.action} | {result.timing_ms['end_to_end']:.1f} ms",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return canvas
