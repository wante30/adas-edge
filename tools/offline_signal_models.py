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

    def detect(self, image):
        started = time.perf_counter()
        rgb, scale, px, py = letterbox_bgr(image, self.size)
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
        rows = decode_outputs([boxes, scores], .001, .45, score_logits=self.logits)
        height, width = image.shape[:2]
        found = []
        for row in rows:
            if row['class_id'] != self.tl_id:
                continue
            x1, y1, x2, y2 = row['bbox']
            box = [float(np.clip((x1-px)/scale, 0, width)), float(np.clip((y1-py)/scale, 0, height)),
                   float(np.clip((x2-px)/scale, 0, width)), float(np.clip((y2-py)/scale, 0, height))]
            if box[2] > box[0] and box[3] > box[1]:
                found.append(dict(bbox=box, confidence=row['confidence'], class_name='traffic light'))
        return found, (time.perf_counter()-started)*1000
