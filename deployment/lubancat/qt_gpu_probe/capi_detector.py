"""ctypes wrappers for the persistent P2 detection / MTL C API backends."""

from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

LIB_PATH = os.environ.get("ADAS_P2CAPI_LIB",
                          "/home/cat/qt_gpu_probe_ai_20261019/libp2capi.so")
OVERLAY_PATH = "/dev/shm/adas_mtl_overlay.png"
OVERLAY_META = "/dev/shm/adas_mtl_overlay.json"
BDD_NAMES = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
             "bicycle", "traffic light", "traffic sign")
COMPETITION_CLASSES = {"bus", "car", "bicycle", "traffic light", "truck",
                       "stop sign", "person", "motorcycle"}

# lane_cat: 0 crosswalk, 1 double other, 2 double white, 3 double yellow,
#           4 road curb, 5 single other, 6 single white, 7 single yellow, 8 bg
# lane_style: 0 solid, 1 dashed, 2 background
# drivable: 0 direct, 1 alternative, 2 background
LANE_BACKGROUND = 8
LANE_CROSSWALK = 0
STYLE_SOLID = 0
STYLE_DASHED = 1
DRIVE_DIRECT = 0
DRIVE_ALTERNATIVE = 1
COLOR_CROSSWALK = (255, 0, 255, 190)
COLOR_SOLID = (0, 0, 255, 200)
COLOR_DASHED = (255, 255, 0, 200)
COLOR_DIRECT = (0, 200, 40, 70)
COLOR_ALTERNATIVE = (255, 96, 0, 70)


class P2Result(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_int),
        ("rknn_ms", ctypes.c_float),
        ("decode_ms", ctypes.c_float),
        ("total_ms", ctypes.c_float),
        ("data", (ctypes.c_float * 6) * 1000),
    ]


def _normalize_items(rows, scale, pad_x, pad_y, class_names):
    """Match the console adapter's class naming and filtering rules."""
    items = []
    for row in rows:
        cls = int(row[0])
        score = float(row[1])
        name = class_names[cls] if 0 <= cls < len(class_names) else f"class_{cls}"
        if name == "pedestrian":
            name = "person"
        elif name in ("motor", "motorcycle"):
            name = "motorcycle"
        elif name in ("bike", "bicycle"):
            name = "bicycle"
        elif name == "traffic sign":
            continue
        if name in ("motorcycle", "bicycle") and score < 0.85:
            continue
        if name not in COMPETITION_CLASSES:
            continue
        items.append({
            "class_name": name,
            "confidence": score,
            "bbox": [(float(row[2]) - pad_x) / scale,
                     (float(row[3]) - pad_y) / scale,
                     (float(row[4]) - pad_x) / scale,
                     (float(row[5]) - pad_y) / scale],
            "source": "MAIN_RKNN",
        })
    return items


class CapiDetector:
    def __init__(self, model_path, input_size=(512, 288), specialist=False,
                 class_names=None, class_profile=None, backend="rknn_capi"):
        if specialist:
            raise ValueError("C API backend is main-model only")
        self.model_path = str(model_path)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.class_names = tuple(class_names or BDD_NAMES)
        self.specialist = False
        self.single_model = True
        self.source = "MAIN_RKNN"
        self.frame_id = 0
        self.task = "detect"
        self.last_timing = None
        self._lib = ctypes.CDLL(LIB_PATH)
        self._lib.p2_create.argtypes = [ctypes.c_char_p]
        self._lib.p2_create.restype = ctypes.c_void_p
        self._lib.p2_detect.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                                        ctypes.POINTER(P2Result)]
        self._lib.p2_detect.restype = ctypes.c_int
        self._lib.p2_destroy.argtypes = [ctypes.c_void_p]
        self._lib.p2_destroy.restype = None
        self._ctx = self._lib.p2_create(str(model_path).encode("utf-8"))
        if not self._ctx:
            raise RuntimeError(f"p2_create failed: {model_path}")
        self.detector = self

    def _rows(self, result):
        return [(result.data[i][0], result.data[i][1], result.data[i][2],
                 result.data[i][3], result.data[i][4], result.data[i][5])
                for i in range(result.count)]

    def detect(self, frame):
        from tools.profile_rknn_single_stream import letterbox_bgr

        t0 = time.perf_counter_ns()
        image, scale, pad_x, pad_y = letterbox_bgr(frame, self.input_size)
        t1 = time.perf_counter_ns()
        result = P2Result()
        ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ret = self._lib.p2_detect(self._ctx, ptr, ctypes.byref(result))
        t2 = time.perf_counter_ns()
        if ret != 0:
            raise RuntimeError(f"p2_detect failed: {ret}")
        items = _normalize_items(self._rows(result), scale, pad_x, pad_y,
                                 self.class_names)
        self.last_timing = {
            "preprocess_ms": (t1 - t0) / 1e6,
            "rknn_call_ms": float(result.rknn_ms),
            "decode_nms_ms": float(result.decode_ms),
            "total_ms": (t2 - t0) / 1e6,
            "output_count": 12,
            "detection_count": len(items),
            "preprocess_backend": "opencv",
        }
        self.frame_id += 1
        return items, float(result.rknn_ms + result.decode_ms)

    def close(self):
        if getattr(self, "_ctx", None):
            self._lib.p2_destroy(self._ctx)
            self._ctx = None


class CapiMtlDetector(CapiDetector):
    def __init__(self, model_path, input_size=(512, 288), specialist=False,
                 class_names=None, class_profile=None, backend="rknn_capi_mtl"):
        super().__init__(model_path, input_size, specialist=specialist,
                         class_names=class_names, class_profile=class_profile,
                         backend=backend)
        self.task = "mtl"
        self.last_masks = None
        self.mask_h = 0
        self.mask_w = 0
        self.overlay_generation = 0
        self.overlay_written = False
        self._writer_cv = threading.Condition()
        self._writer_stop = False
        self._writer_seq = 0
        self._writer_written = 0
        self._writer_pending = None
        self.last_overlay_ms = None
        self._writer = threading.Thread(target=self._overlay_worker,
                                        name="mtl-overlay-writer", daemon=True)
        self._writer.start()
        self._lib.p2_seg_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int)]
        self._lib.p2_seg_size.restype = None
        self._lib.p2_detect_mtl.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(P2Result), ctypes.POINTER(ctypes.c_uint8)]
        self._lib.p2_detect_mtl.restype = ctypes.c_int
        h = ctypes.c_int()
        w = ctypes.c_int()
        self._lib.p2_seg_size(self._ctx, ctypes.byref(h), ctypes.byref(w))
        self.mask_h, self.mask_w = int(h.value), int(w.value)
        if self.mask_h <= 0 or self.mask_w <= 0:
            self.close()
            raise RuntimeError("model has no segmentation outputs")

    def detect(self, frame):
        from tools.profile_rknn_single_stream import letterbox_bgr

        t0 = time.perf_counter_ns()
        image, scale, pad_x, pad_y = letterbox_bgr(frame, self.input_size)
        height, width = frame.shape[:2]
        resized_w = int(round(width * scale))
        resized_h = int(round(height * scale))
        t1 = time.perf_counter_ns()
        result = P2Result()
        masks = np.zeros((3, self.mask_h, self.mask_w), dtype=np.uint8)
        ret = self._lib.p2_detect_mtl(
            self._ctx,
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(result),
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        t2 = time.perf_counter_ns()
        if ret != 0:
            raise RuntimeError(f"p2_detect_mtl failed: {ret}")
        self.last_masks = masks
        items = _normalize_items(self._rows(result), scale, pad_x, pad_y,
                                 self.class_names)
        inference_ms = float(result.rknn_ms + result.decode_ms)
        self.last_timing = {
            "preprocess_ms": (t1 - t0) / 1e6,
            "rknn_call_ms": float(result.rknn_ms),
            "decode_nms_ms": float(result.decode_ms),
            "seg_ms": max(0.0, (t2 - t1) / 1e6 - float(result.rknn_ms)
                          - float(result.decode_ms)),
            "total_ms": (t2 - t0) / 1e6,
            "output_count": 15,
            "detection_count": len(items),
            "preprocess_backend": "opencv",
        }
        self.frame_id += 1
        self._queue_overlay(pad_x, pad_y, resized_w, resized_h)
        self.last_timing["overlay_ms"] = 0.0
        self.last_timing["overlay_encode_ms"] = self.last_overlay_ms
        return items, inference_ms

    def _queue_overlay(self, pad_x, pad_y, resized_w, resized_h):
        y0 = max(0, int(pad_y))
        x0 = max(0, int(pad_x))
        y1 = min(self.mask_h, y0 + int(resized_h))
        x1 = min(self.mask_w, x0 + int(resized_w))
        masks = self.last_masks
        if masks is None:
            return
        lane = masks[0][y0:y1, x0:x1]
        style = masks[1][y0:y1, x0:x1]
        drivable = masks[2][y0:y1, x0:x1]
        if lane.size == 0:
            return
        with self._writer_cv:
            self._writer_pending = (lane, style, drivable, int(self.frame_id),
                                    time.monotonic_ns())
            self._writer_seq += 1
            self._writer_cv.notify()

    def _overlay_worker(self):
        while True:
            with self._writer_cv:
                while not self._writer_stop and self._writer_seq == self._writer_written:
                    self._writer_cv.wait()
                if self._writer_stop:
                    return
                job = self._writer_pending
                sequence = self._writer_seq
                self._writer_pending = None
            if job is None:
                self._writer_written = sequence
                continue
            started = time.perf_counter()
            try:
                self._write_overlay(*job)
            except Exception:
                pass
            self.last_overlay_ms = (time.perf_counter() - started) * 1000
            self._writer_written = sequence

    def _write_overlay(self, lane, style, drivable, frame_id, timestamp_ns):
        height, width = lane.shape
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        canvas[drivable == DRIVE_DIRECT] = COLOR_DIRECT
        canvas[drivable == DRIVE_ALTERNATIVE] = COLOR_ALTERNATIVE
        foreground = lane != LANE_BACKGROUND
        canvas[foreground & (style == STYLE_DASHED)] = COLOR_DASHED
        canvas[foreground & (style != STYLE_DASHED)
               & (lane != LANE_CROSSWALK)] = COLOR_SOLID
        canvas[lane == LANE_CROSSWALK] = COLOR_CROSSWALK
        ok, encoded = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            return
        tmp = OVERLAY_PATH + ".tmp"
        with open(tmp, "wb") as stream:
            stream.write(encoded.tobytes())
        os.replace(tmp, OVERLAY_PATH)
        self.overlay_generation += 1
        self.overlay_written = True
        meta = {
            "frame_id": int(frame_id),
            "timestamp_ns": int(timestamp_ns),
            "mask_width": int(self.mask_w),
            "mask_height": int(self.mask_h),
            "display_width": int(width),
            "display_height": int(height),
            "generation": int(self.overlay_generation),
            "model_path": Path(self.model_path).name,
        }
        tmp_json = OVERLAY_META + ".tmp"
        with open(tmp_json, "w") as stream:
            json.dump(meta, stream)
        os.replace(tmp_json, OVERLAY_META)

    def close(self):
        with self._writer_cv:
            self._writer_stop = True
            self._writer_cv.notify()
        self._writer.join(timeout=2.0)
        if self.overlay_written:
            for path in (OVERLAY_PATH, OVERLAY_META):
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self.overlay_written = False
        super().close()
