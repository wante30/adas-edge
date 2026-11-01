from __future__ import annotations
import time
from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
from .types import Detection, FrameResult

COCO_NAMES = ("person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven"," toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush")
BDD_NAMES = ("pedestrian","rider","car","truck","bus","train","motorcycle","bicycle","traffic light","traffic sign")
COMPETITION_CLASSES = {"person","bicycle","car","motorcycle","bus","truck","traffic light","stop sign"}
MODEL_CLASS_PROFILES = {"coco80": COCO_NAMES, "bdd10": BDD_NAMES}
TIMING_HISTORY_LIMIT = 3600

class CompetitionDetector:
    def __init__(self, model_path: str | Path, input_size: int | tuple[int, int] = 320, conf_threshold: float = .25, nms_threshold: float = .45, runner: Any = None, class_names=None, class_profile: str | None = None, use_rga_preprocess: bool = False):
        self.model_path = Path(model_path)
        if class_names is not None:
            self.class_names = tuple(str(name) for name in class_names)
            if not self.class_names or len(set(self.class_names)) != len(self.class_names):
                raise ValueError("class_names must be non-empty and unique")
        elif class_profile is not None:
            try:
                self.class_names = MODEL_CLASS_PROFILES[class_profile]
            except KeyError as exc:
                raise ValueError(f"unknown model profile: {class_profile!r}") from exc
        elif "bdd" in self.model_path.name.lower():
            self.class_names = BDD_NAMES
        else:
            self.class_names = COCO_NAMES
        self.input_size = input_size if isinstance(input_size, int) else (int(input_size[0]), int(input_size[1]))
        self.conf_threshold, self.nms_threshold = float(conf_threshold), float(nms_threshold)
        self._runner = runner
        self.use_rga_preprocess = bool(use_rga_preprocess)
        self.last_preprocess_backend = "opencv"
        self.timing_history = deque(maxlen=TIMING_HISTORY_LIMIT)
        self.last_timing = None
        if runner is None:
            from rknnlite.api import RKNNLite
            self._runner = RKNNLite(verbose=False)
            if self._runner.load_rknn(str(self.model_path)) != 0 or self._runner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                raise RuntimeError(f"failed to initialize RKNN model: {self.model_path}")

    def preprocess(self, frame: np.ndarray):
        from tools.profile_rknn_single_stream import letterbox_bgr
        if getattr(self, "use_rga_preprocess", False):
            from .rga_preprocess import preprocess_bgr_rga
            rga_result = preprocess_bgr_rga(frame, self.input_size)
            if rga_result is not None:
                self.last_preprocess_backend = "rga"
                return rga_result
        self.last_preprocess_backend = "opencv"
        return letterbox_bgr(frame, self.input_size)

    def infer(self, frame: np.ndarray, stream_id: int | str = 0, frame_id: int = 0, timestamp_ns: int | None = None) -> FrameResult:
        from deployment.rknn.decode_yolo_output import decode_outputs
        from deployment.rknn.decode_raw_heads import (
            decode_raw_heads_thresholded,
            looks_like_raw_head_triplets,
        )
        from tools.profile_rknn_single_stream import decode_yolo11_raw
        preprocess_started = time.perf_counter_ns()
        image, scale, px, py = self.preprocess(frame)
        preprocess_ms = (time.perf_counter_ns() - preprocess_started) / 1e6
        rknn_started = time.perf_counter_ns()
        outputs = self._runner.inference(inputs=[image[None, ...]], data_format="nhwc")
        rknn_call_ms = (time.perf_counter_ns() - rknn_started) / 1e6
        postprocess_started = time.perf_counter_ns()
        if len(outputs) == 2:
            decoded = decode_outputs(outputs, self.conf_threshold, self.nms_threshold)
            detections = [[*d["bbox"], d["confidence"], d["class_id"]] for d in decoded]
            # split decoder emits model-space coordinates; restore to source frame
            for row in detections:
                row[0], row[2] = (row[0] - px) / scale, (row[2] - px) / scale
                row[1], row[3] = (row[1] - py) / scale, (row[3] - py) / scale
        elif looks_like_raw_head_triplets(outputs):
            detections = decode_raw_heads_thresholded(
                outputs,
                input_width=int(self.input_size[0]),
                input_height=int(self.input_size[1]),
                confidence=self.conf_threshold,
                iou=self.nms_threshold,
                scale=scale,
                pad_x=px,
                pad_y=py,
            )
        else:
            detections = decode_yolo11_raw(outputs, self.input_size, self.conf_threshold, self.nms_threshold, scale, px, py)
        postprocess_ms = (time.perf_counter_ns() - postprocess_started) / 1e6
        inference_ms = rknn_call_ms + postprocess_ms
        elapsed = preprocess_ms + inference_ms
        timing = {
            "preprocess_ms": preprocess_ms,
            "rknn_call_ms": rknn_call_ms,
            "decode_nms_ms": postprocess_ms,
            "total_ms": elapsed,
            "output_count": len(outputs),
            "detection_count": len(detections),
            "preprocess_backend": getattr(self, "last_preprocess_backend", "opencv"),
        }
        self.last_timing = timing
        history = getattr(self, "timing_history", None)
        if history is None:
            history = deque(maxlen=TIMING_HISTORY_LIMIT)
            self.timing_history = history
        history.append(timing)
        names = getattr(self, "class_names", COCO_NAMES)
        result = [Detection(int(row[5]), names[int(row[5])] if 0 <= int(row[5]) < len(names) else f"class_{int(row[5])}", float(row[4]), *map(float, row[:4])) for row in detections]
        return FrameResult(stream_id, int(frame_id), timestamp_ns, inference_ms, result)

    def close(self):
        if self._runner is not None and hasattr(self._runner, "release"):
            self._runner.release()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
