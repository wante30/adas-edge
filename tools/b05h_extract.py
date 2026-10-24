"""B-05H extract de-duplicated competition-road frames with provenance."""
from pathlib import Path
import cv2
import numpy as np
import json

ROOT = Path(r"D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材")
OUT = Path(r"C:\Users\JingyuWang\adas-edge\outputs\b05h")
VIDEOS = {
    "video_20260929_113421.mp4": (0.0, 13.0, 2.0),
    "video_20260929_112507.mp4": (0.0, 20.0, 1.0),
    "video_20260929_113539.mp4": (0.0, 20.0, 1.0),
    "video_20260929_112739.mp4": (0.0, 15.0, 1.0),
}


def dhash(image, size=16):
    gray = cv2.cvtColor(cv2.resize(image, (size + 1, size)), cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return np.packbits(diff.reshape(-1)).tobytes()


def hamming(a, b):
    return sum(int(x).bit_count() for x in bytes(x ^ y for x, y in zip(a, b)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (start, end, step) in VIDEOS.items():
        cap = cv2.VideoCapture(str(ROOT / name))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        source_dir = OUT / "frames" / name.replace(".mp4", "")
        source_dir.mkdir(parents=True, exist_ok=True)
        seen = []
        t = start
        saved = 0
        while t < min(end, total / fps):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            image = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            digest = dhash(image)
            if any(hamming(digest, old) <= 4 for old in seen):
                t += step
                continue
            seen.append(digest)
            frame_id = int(round(t * fps))
            path = source_dir / f"{name.replace('.mp4','')}_{t:06.2f}_{frame_id:06d}.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({
                "image": str(path),
                "video": name,
                "timestamp_s": round(float(t), 3),
                "source_frame_id": frame_id,
                "width": 1280,
                "height": 720,
                "source": "competition_road_video",
            })
            saved += 1
            t += step
        cap.release()
        print(name, "saved", saved)
    (OUT / "frames_index.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("TOTAL", len(rows))


if __name__ == "__main__":
    main()
