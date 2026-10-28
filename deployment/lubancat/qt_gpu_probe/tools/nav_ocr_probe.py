#!/usr/bin/env python3
"""Diagnostic OCR for the current Q2 navigation screen (top-right quadrant)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

full = cv2.imread(str(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/nav_current.png")))
if full is None:
    raise SystemExit("cannot read image")
h, w = full.shape[:2]
# Q2 is the top-right quadrant of the 2x2 PCIe frame.
q2 = full[0:h // 2, w // 2:w]
cv2.imwrite("/tmp/q2_nav.png", q2)
# Route card region in the native Q2 640x360 image.
card = q2[70:190, 20:235]
card = cv2.resize(card, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("/tmp/q2_nav_card.png", card)

from rapidocr import RapidOCR

engine = RapidOCR(params={"Global.log_level": "error"})
for name, image in (("Q2", cv2.resize(q2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
                    ("CARD", card)):
    result = engine(image, use_det=True, use_cls=True)
    print(f"--- {name} shape={image.shape}")
    txts = list(result.txts or ())
    scores = list(result.scores or ())
    boxes = list(result.boxes) if result.boxes is not None else []
    for text, score, box in zip(txts, scores, boxes):
        print(f"{score:.3f}\t{text}\t{box}")
