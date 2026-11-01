"""Frozen split-output ONNX detector for PC tests, not an RKNN simulator."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from competition_ai.detector import COCO_NAMES, COMPETITION_CLASSES
from deployment.rknn.decode_yolo_output import decode_outputs
from tools.profile_rknn_single_stream import letterbox_bgr


class PcDetector:
    def __init__(self, path: Path, input_size, *, specialist=False):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        meta = self.session.get_inputs()[0]
        if meta.shape != [1,3,input_size[1],input_size[0]]:
            raise ValueError(f"Model input does not match frozen shape: {meta.shape}")
        self.name, self.size, self.specialist = meta.name, input_size, specialist

    def detect(self, frame):
        started = time.perf_counter()
        rgb,scale,px,py = letterbox_bgr(frame,self.size)
        tensor = np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)/255.
        output = self.session.run(None,{self.name:tensor})
        decoded = decode_outputs(output,.05 if self.specialist else .25,.70 if self.specialist else .45)
        height,width = frame.shape[:2]
        items = []
        for item in decoded:
            cid = item['class_id']
            if self.specialist and cid != 0:
                raise ValueError("Specialist has an unexpected class")
            name = 'traffic light' if self.specialist else COCO_NAMES[cid]
            if name not in COMPETITION_CLASSES:
                continue
            x1,y1,x2,y2 = item['bbox']
            box = [float(np.clip((x1-px)/scale,0,width)),float(np.clip((y1-py)/scale,0,height)),
                   float(np.clip((x2-px)/scale,0,width)),float(np.clip((y2-py)/scale,0,height))]
            if box[2] > box[0] and box[3] > box[1]:
                items.append({'class_name':name,'confidence':item['confidence'],'bbox':box,
                              'source':'TL_SPECIALIST_ONNX_PC' if self.specialist else 'MAIN_ONNX_PC'})
        return items,(time.perf_counter()-started)*1000
