import json
import os
import cv2

VIDEO = r"D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112507.mp4"
OUT = r"D:\Datasets\adas_edge_training\b05n_mtl_demo_20261031\ai_jpg"
os.makedirs(OUT, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, int(round(fps / 6.0)))
indices = list(range(0, count, step))
print("fps", fps, "frames", count, "step", step, "samples", len(indices), flush=True)

saved = []
for i, idx in enumerate(indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        print("read fail", idx, flush=True)
        continue
    img = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    path = os.path.join(OUT, f"{i:04d}.jpg")
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    saved.append({"sample": i, "frame": idx, "file": os.path.basename(path)})
cap.release()
meta = {"fps": fps, "frames": count, "step": step, "samples": len(saved), "items": saved}
open(os.path.join(OUT, "..", "ai_sample_meta.json"), "w").write(json.dumps(meta, indent=2))
print("saved", len(saved), "to", OUT)
