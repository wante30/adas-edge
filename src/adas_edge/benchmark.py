from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import psutil

from adas_edge.pipeline import PipelineResult


CSV_COLUMNS = [
    "camera_id", "frame_id", "model", "input_size", "precision", "effective_inference_fps",
    "preprocess_latency_ms", "inference_latency_ms", "postprocess_latency_ms",
    "end_to_end_latency_ms", "queue_latency_ms", "dropped_frames", "memory_mb",
]


@dataclass
class BenchmarkRecorder:
    model: str
    input_size: int
    precision: str
    _previous_end_to_end_ms: float | None = None

    def row(self, result: PipelineResult, queue_latency_ms: float = 0.0, dropped_frames: int = 0) -> dict[str, float | int | str]:
        end_to_end = result.timing_ms["end_to_end"]
        inference_fps = 1000 / end_to_end if end_to_end > 0 else 0.0
        self._previous_end_to_end_ms = end_to_end
        return {
            "camera_id": result.frame.camera_id,
            "frame_id": result.frame.frame_id,
            "model": self.model,
            "input_size": self.input_size,
            "precision": self.precision,
            "effective_inference_fps": round(inference_fps, 3),
            "preprocess_latency_ms": result.timing_ms["preprocess"],
            "inference_latency_ms": result.timing_ms["inference"],
            "postprocess_latency_ms": result.timing_ms["postprocess"],
            "end_to_end_latency_ms": end_to_end,
            "queue_latency_ms": queue_latency_ms,
            "dropped_frames": dropped_frames,
            "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 3),
        }

    @staticmethod
    def append_csv(path: str | Path, row: dict[str, float | int | str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
