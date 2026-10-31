"""Run the board's AiReplayBridge with the B-2a 15-output MTL model."""

import importlib
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path("/home/cat/competition_cockpit_port_20261018")
BRIDGE_DIR = Path("/home/cat/qt_gpu_probe_ai_20261019")
BENCH = Path("/home/cat/b05n_mtl_b2a_bench_20261031")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BRIDGE_DIR))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bridge = load_module("b05p_bridge_mtl", BRIDGE_DIR / "b05p_competition_bridge.py")
original_install = bridge._install_raw_head_adapter


def mtl_infer(self, frame, stream_id=0, frame_id=0, timestamp_ns=None):
    from competition_ai.types import Detection, FrameResult
    from deployment.rknn.decode_raw_heads import decode_raw_heads_thresholded

    t0 = time.perf_counter_ns()
    image, scale, px, py = self.preprocess(frame)
    pre_ms = (time.perf_counter_ns() - t0) / 1e6
    t1 = time.perf_counter_ns()
    outputs = self._runner.inference(inputs=[image[None, ...]], data_format="nhwc")
    rknn_ms = (time.perf_counter_ns() - t1) / 1e6
    t2 = time.perf_counter_ns()
    det_outputs = outputs[:12] if len(outputs) >= 15 else outputs
    rows = decode_raw_heads_thresholded(
        det_outputs, input_width=int(self.input_size[0]), input_height=int(self.input_size[1]),
        confidence=self.conf_threshold, iou=self.nms_threshold, scale=scale, pad_x=px, pad_y=py)
    det_ms = (time.perf_counter_ns() - t2) / 1e6
    seg_ms = 0.0
    if len(outputs) >= 15:
        t3 = time.perf_counter_ns()
        for out in outputs[12:15]:
            np.asarray(out)[0].argmax(0)
        seg_ms = (time.perf_counter_ns() - t3) / 1e6
    total_ms = pre_ms + rknn_ms + det_ms + seg_ms
    self.last_timing = {"preprocess_ms": pre_ms, "rknn_call_ms": rknn_ms,
                        "decode_nms_ms": det_ms, "seg_ms": seg_ms, "total_ms": total_ms,
                        "output_count": len(outputs), "detection_count": len(rows),
                        "preprocess_backend": getattr(self, "last_preprocess_backend", "opencv")}
    history = getattr(self, "timing_history", None)
    if history is None:
        history = deque(maxlen=3600)
        self.timing_history = history
    history.append(self.last_timing)
    detections = [Detection(int(r[5]),
                            self.class_names[int(r[5])] if 0 <= int(r[5]) < len(self.class_names) else f"class_{int(r[5])}",
                            float(r[4]), *map(float, r[:4])) for r in rows]
    return FrameResult(stream_id, int(frame_id), timestamp_ns, rknn_ms + det_ms + seg_ms, detections)


def install(m, scheme):
    original_install(m, scheme)
    import competition_ai.detector as detector
    detector.CompetitionDetector.infer = mtl_infer


bridge._install_raw_head_adapter = install

args = SimpleNamespace(
    scheme="B",
    bridge_script=BRIDGE_DIR / "ai_bridge.py",
    competition_root=ROOT,
    data_root=BENCH,
    model_dir=BENCH / "models",
    replay_dir=BENCH / "replay60",
    output_dir=BENCH / "bridge_b2a",
    frame_count=60,
    warmup=10,
    requests=180,
    request_hz=10.0,
)
report = bridge.run_benchmark(args)
print(json.dumps({
    "fps": report["measurement"]["measured_result_fps"],
    "analyzed": report["measurement"]["measured_frames_analyzed"],
    "replaced": report["measurement"]["requests_replaced_latest_only"],
    "main": report["measurement"]["main_model_timing"],
    "npu": report["measurement"]["main_npu_timing"],
    "det": report["measurement"]["main_decode_nms_timing"],
    "pipeline": report["measurement"]["bridge_pipeline_timing"],
}, indent=2))
