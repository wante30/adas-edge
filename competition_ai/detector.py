from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import numpy as np
from .types import Detection, FrameResult

COCO_NAMES = ("person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven"," toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush")
COMPETITION_CLASSES = {"person","bicycle","car","motorcycle","bus","truck","traffic light","stop sign"}

class CompetitionDetector:
    def __init__(self, model_path: str | Path, input_size: int = 320, conf_threshold: float = .25, nms_threshold: float = .45, runner: Any = None):
        self.model_path, self.input_size = Path(model_path), int(input_size)
        self.conf_threshold, self.nms_threshold = float(conf_threshold), float(nms_threshold)
        self._runner = runner
        if runner is None:
            from rknnlite.api import RKNNLite
            self._runner = RKNNLite(verbose=False)
            if self._runner.load_rknn(str(self.model_path)) != 0 or self._runner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                raise RuntimeError(f"failed to initialize RKNN model: {self.model_path}")

    def preprocess(self, frame: np.ndarray):
        from tools.profile_rknn_single_stream import letterbox_bgr
        return letterbox_bgr(frame, self.input_size)

    def infer(self, frame: np.ndarray, stream_id: int | str = 0, frame_id: int = 0, timestamp_ns: int | None = None) -> FrameResult:
        from deployment.rknn.decode_yolo_output import decode_outputs
        from tools.profile_rknn_single_stream import decode_yolo11_raw
        image, scale, px, py = self.preprocess(frame)
        started = time.perf_counter()
        outputs = self._runner.inference(inputs=[image[None, ...]], data_format="nhwc")
        if len(outputs) == 2:
            decoded = decode_outputs(outputs, self.conf_threshold, self.nms_threshold)
            detections = [[*d["bbox"], d["confidence"], d["class_id"]] for d in decoded]
            # split decoder emits model-space coordinates; restore to source frame
            for row in detections:
                row[0], row[2] = (row[0] - px) / scale, (row[2] - px) / scale
                row[1], row[3] = (row[1] - py) / scale, (row[3] - py) / scale
        else:
            detections = decode_yolo11_raw(outputs, self.input_size, self.conf_threshold, self.nms_threshold, scale, px, py)
        elapsed = (time.perf_counter() - started) * 1000
        result = [Detection(int(row[5]), COCO_NAMES[int(row[5])] if 0 <= int(row[5]) < len(COCO_NAMES) else f"class_{int(row[5])}", float(row[4]), *map(float, row[:4])) for row in detections]
        return FrameResult(stream_id, int(frame_id), timestamp_ns, elapsed, result)

    def close(self):
        if self._runner is not None and hasattr(self._runner, "release"):
            self._runner.release()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
