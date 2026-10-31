import glob
import os
import cv2
import numpy as np

SRC = "/home/cat/b05n_mtl_b2a_demo_20261031/ai_jpg"
OUT = "/home/cat/b05n_mtl_b2a_demo_20261031/inputs.bin"
files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
print("files", len(files))
with open(OUT, "wb") as f:
    for p in files:
        img = cv2.imread(p)
        if img is None:
            raise SystemExit("read fail " + p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape != (288, 512, 3):
            raise SystemExit(f"bad shape {img.shape} {p}")
        f.write(np.ascontiguousarray(img).tobytes())
print("bytes", os.path.getsize(OUT), "frames", len(files))
