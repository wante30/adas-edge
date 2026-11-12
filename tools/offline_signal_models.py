"""PC float ONNX counterparts of the deployed detectors; no hardware imports."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from deployment.rknn.decode_yolo_output import decode_outputs
from tools.profile_rknn_single_stream import letterbox_bgr


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class OfflineDetector:
    def __init__(self, path, *, logits=False, traffic_light_id=None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        meta = self.session.get_inputs()[0]
        self.input_name = meta.name
        self.size = (meta.shape[3], meta.shape[2])
        names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map.get('names', '{}'))
        inferred = [int(k) for k, v in names.items() if v == 'traffic light']
        if traffic_light_id is None and len(inferred) != 1:
            raise ValueError('Supply verified traffic_light_id for graph without class metadata')
        self.tl_id = inferred[0] if traffic_light_id is None else traffic_light_id
        if inferred and self.tl_id != inferred[0]:
            raise ValueError('Class ID contradicts model metadata')
        self.logits = logits
        self.evidence = dict(path=str(Path(path).resolve()), sha256=sha256(path),
            input_size=self.size, traffic_light_id=self.tl_id, class_names=names,
            precision='ONNX_FLOAT_PC_NOT_RKNN_INT8', provider='CPUExecutionProvider',
            scores_are_logits=logits, diagnostic_score_floor=.001, nms_iou=.45)

    @staticmethod
    def _trace_rows(boxes, scores, *, score_floor, nms_iou, score_logits):
        """Return pre-NMS and post-NMS rows in model coordinates."""
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        xywh = boxes[0].T
        xyxy = np.column_stack([
            xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
            xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2,
        ])
        class_scores = scores[0].T
        if score_logits:
            class_scores = 1.0 / (1.0 + np.exp(-class_scores))
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        pre, post, suppressed = [], [], []
        for class_id in np.unique(class_ids):
            valid = np.flatnonzero((class_ids == class_id) & (confidences >= score_floor))
            if not len(valid):
                continue
            rows = [{"class_id": int(class_ids[index]), "confidence": float(confidences[index]),
                     "bbox_model": [float(value) for value in xyxy[index]], "source_index": int(index)}
                    for index in valid]
            pre.extend(rows)
            order = confidences[valid].argsort()[::-1]
            keep = []
            remaining = list(valid[order])
            while remaining:
                current = remaining.pop(0)
                keep.append(int(current))
                survivors = []
                for candidate in remaining:
                    a, b = xyxy[current], xyxy[candidate]
                    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                    intersection = max(0., ix2-ix1) * max(0., iy2-iy1)
                    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
                    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
                    overlap = intersection / max(area_a + area_b - intersection, 1e-6)
                    if overlap <= nms_iou:
                        survivors.append(candidate)
                    else:
                        suppressed.append({"class_id": int(class_ids[candidate]),
                                           "confidence": float(confidences[candidate]),
                                           "bbox_model": [float(value) for value in xyxy[candidate]],
                                           "suppressed_by_source_index": int(current),
                                           "iou": float(overlap)})
                remaining = survivors
            keep_set = set(keep)
            post.extend(row for row in rows if row["source_index"] in keep_set)
        pre.sort(key=lambda row: row["confidence"], reverse=True)
        post.sort(key=lambda row: row["confidence"], reverse=True)
        suppressed.sort(key=lambda row: row["confidence"], reverse=True)
        return pre, post, suppressed

    @staticmethod
    def _map_trace(rows, *, scale, px, py, width, height):
        mapped = []
        for row in rows:
            x1, y1, x2, y2 = row["bbox_model"]
            box = [float(np.clip((x1-px)/scale, 0, width)), float(np.clip((y1-py)/scale, 0, height)),
                   float(np.clip((x2-px)/scale, 0, width)), float(np.clip((y2-py)/scale, 0, height))]
            mapped.append({**row, "bbox": box})
        return mapped

    def detect(self, image, *, return_trace=False, color_order="rgb"):
        started = time.perf_counter()
        rgb, scale, px, py = letterbox_bgr(image, self.size)
        if color_order not in {"rgb", "bgr"}:
            raise ValueError("color_order must be rgb or bgr")
        if color_order == "bgr":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tensor = np.transpose(rgb.astype(np.float32)/255, (2, 0, 1))[None]
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) == 1:  # Ultralytics decoded [xywh, class probabilities]
            boxes, scores = outputs[0][:, :4], outputs[0][:, 4:]
        elif len(outputs) == 2:
            boxes, scores = outputs
        else:  # MTL raw DFL branches; segmentation outputs are not detections.
            decoded, probabilities = [], []
            names = [o.name for o in self.session.get_outputs()]
            for name, reg in zip(names, outputs):
                if not name.startswith('reg_s'):
                    continue
                stride = int(name.split('_s')[1])
                cls = outputs[names.index('cls_s'+str(stride))]
                _, _, h, w = reg.shape
                distribution = reg.reshape(1, 4, 16, h*w)
                distribution = np.exp(distribution-distribution.max(axis=2, keepdims=True))
                distance = (distribution/distribution.sum(axis=2, keepdims=True)*np.arange(16)[None, None, :, None]).sum(axis=2)
                yy, xx = np.mgrid[:h, :w]
                center = np.stack([xx.ravel()+.5, yy.ravel()+.5])[None]
                lo, hi = center-distance[:, :2], center+distance[:, 2:]
                decoded.append(np.concatenate([(lo+hi)/2, hi-lo], axis=1)*stride)
                probabilities.append(cls.reshape(1, cls.shape[1], -1))
            boxes, scores = np.concatenate(decoded, axis=2), np.concatenate(probabilities, axis=2)
        # Class-aware decoder, then filter by verified metadata. Preserve low
        # confidence candidates for threshold diagnostics, never promote them.
        pre_rows, post_rows, suppressed_rows = self._trace_rows(
            boxes, scores, score_floor=.001, nms_iou=.45, score_logits=self.logits)
        height, width = image.shape[:2]
        found = []
        for row in post_rows:
            if row['class_id'] != self.tl_id:
                continue
            x1, y1, x2, y2 = row['bbox_model']
            box = [float(np.clip((x1-px)/scale, 0, width)), float(np.clip((y1-py)/scale, 0, height)),
                   float(np.clip((x2-px)/scale, 0, width)), float(np.clip((y2-py)/scale, 0, height))]
            if box[2] > box[0] and box[3] > box[1]:
                found.append(dict(bbox=box, confidence=row['confidence'], class_name='traffic light'))
        elapsed = (time.perf_counter()-started)*1000
        if not return_trace:
            return found, elapsed
        trace = {
            "input_image_size": [int(width), int(height)],
            "model_input_size": [int(self.size[0]), int(self.size[1])],
            "letterbox": {"scale": float(scale), "pad_x": int(px), "pad_y": int(py),
                           "color_order": color_order, "normalization": "RGB/255"},
            "score_floor": .001,
            "nms_iou": .45,
            "traffic_light_id": int(self.tl_id),
            "pre_nms": self._map_trace([r for r in pre_rows if r["class_id"] == self.tl_id],
                                        scale=scale, px=px, py=py, width=width, height=height),
            "post_nms": self._map_trace([r for r in post_rows if r["class_id"] == self.tl_id],
                                         scale=scale, px=px, py=py, width=width, height=height),
            "nms_suppressed": self._map_trace([r for r in suppressed_rows if r["class_id"] == self.tl_id],
                                               scale=scale, px=px, py=py, width=width, height=height),
            "elapsed_ms": elapsed,
        }
        return found, elapsed, trace
