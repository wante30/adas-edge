"""Direct model adapters for rectangular PyTorch and ONNX Runtime inference."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def resolve_model_path(
    *,
    backend: str,
    model_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    if (model_path is None) == (manifest_path is None):
        raise ValueError("provide exactly one of model_path or manifest_path")
    if model_path is not None:
        return Path(model_path).expanduser().resolve()

    manifest = Path(manifest_path).expanduser().resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    hash_key = "weights_best_pt" if backend == "pytorch" else "onnx_static"
    expected = str(data.get("sha256", {}).get(hash_key, "")).lower()
    if len(expected) != 64:
        raise ValueError(f"manifest has no valid sha256.{hash_key}: {manifest}")
    suffix = ".pt" if backend == "pytorch" else ".onnx"
    matches = []
    for candidate in manifest.parent.rglob(f"*{suffix}"):
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() == expected:
            matches.append(candidate.resolve())
    if not matches:
        raise FileNotFoundError(
            f"manifest hash {expected} matched no {suffix} files under {manifest.parent}"
        )
    return min(matches, key=lambda path: (len(path.relative_to(manifest.parent).parts), str(path).lower()))


class ModelRuntime:
    def __init__(self, backend: str, model_path: str | Path, *, device: str = "auto"):
        self.backend = backend
        self.model_path = str(model_path)
        self.device = "cpu"
        self.providers: list[str] = []
        if backend == "pytorch":
            import torch
            from ultralytics import YOLO

            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"PyTorch CUDA device requested but CUDA is unavailable: {device}")
            self.device = device
            self.torch = torch
            self.model = YOLO(self.model_path).model.to(device).eval()
            self.providers = [f"PyTorch:{device}"]
        elif backend == "onnx":
            import onnxruntime as ort

            available = ort.get_available_providers()
            providers = []
            if device != "cpu" and "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input = self.session.get_inputs()[0]
            self.output = self.session.get_outputs()[0]
            self.device = self.session.get_providers()[0]
            self.providers = self.session.get_providers()
            if self.input.shape[:2] != [1, 3] or len(self.input.shape) != 4:
                raise ValueError(f"unexpected ONNX input shape: {self.input.shape}")
            if len(self.output.shape) != 3 or self.output.shape[0] != 1 or self.output.shape[1] != 5:
                raise ValueError(f"expected single-class YOLO11 [1,5,N] output, got {self.output.shape}")
        else:
            raise ValueError(f"unsupported backend: {backend}")

    def validate_canvas(self, width: int, height: int) -> None:
        if self.backend == "onnx":
            expected = self.input.shape
            if expected[2] != height or expected[3] != width:
                raise ValueError(
                    f"ONNX model is fixed at {expected[3]}x{expected[2]}, requested {width}x{height}"
                )

    def run(self, tensor: np.ndarray) -> tuple[Any, float]:
        if self.backend == "pytorch":
            torch = self.torch
            input_tensor = torch.from_numpy(tensor).to(self.device)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.model(input_tensor)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output, elapsed_ms

        started = time.perf_counter()
        output = self.session.run([self.output.name], {self.input.name: tensor})[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return output, elapsed_ms

    def warmup(self, tensor: np.ndarray, runs: int = 2) -> None:
        for _ in range(max(0, runs)):
            self.run(tensor)
