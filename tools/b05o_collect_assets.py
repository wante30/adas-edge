"""Collect B-05O report assets (keyframes + metrics) into the repo."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "research" / "rk3566_models" / "reports" / "assets" / "B05O"
KEY = ASSETS / "keyframes"
BASE = Path(r"D:\Datasets\adas_edge_training\b05o_video_scheduling")
SCHEMES = ("bdd_10hz", "p2_10hz", "dual_10_5hz")


def grab(video: Path, seconds: float, output: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(seconds * fps)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {video} at {seconds}s")
    cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 93])


def main() -> None:
    KEY.mkdir(parents=True, exist_ok=True)
    pairs = {
        "metrics_113421_scheduled.json": BASE / "video_20260929_113421" / "scheduled" / "metrics.json",
        "metrics_112739_scheduled.json": BASE / "video_20260929_112739" / "scheduled" / "metrics.json",
        "metrics_112739_upper.json": BASE / "video_20260929_112739" / "upper_bound" / "metrics.json",
        "metrics_112454_scheduled.json": BASE / "video_20260929_112454" / "scheduled" / "metrics.json",
        "metrics_112454_upper.json": BASE / "video_20260929_112454" / "upper_bound" / "metrics.json",
        "reference_compare_112739.json": BASE / "video_20260929_112739" / "reference_compare.json",
    }
    for name, source in pairs.items():
        if source.exists():
            data = json.loads(source.read_text(encoding="utf-8"))
            if name.startswith("metrics_") and "events" in data:
                data.pop("events")  # keep assets small; full metrics stay on D:
            (ASSETS / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            print("missing", source)

    for seconds, tag in ((81.233, "112739_81.233"), (12.867, "112739_12.867")):
        for scheme in SCHEMES:
            source = BASE / "video_20260929_112739" / "scheduled" / f"{scheme}_t{seconds:07.3f}.jpg"
            if source.exists():
                shutil.copy2(source, KEY / f"{tag}_{scheme}.jpg")
    for scheme in SCHEMES:
        source = BASE / "video_20260929_112739" / "upper_bound" / f"{scheme}_t081.233.jpg"
        if source.exists():
            shutil.copy2(source, KEY / f"112739_81.233_upper_{scheme}.jpg")
        source = BASE / "video_20260929_112454" / "scheduled" / f"{scheme}_t001.500.jpg"
        if source.exists():
            shutil.copy2(source, KEY / f"112454_01.500_{scheme}.jpg")

    for scheme in SCHEMES:
        video = BASE / "video_20260929_112739" / "scheduled" / f"{scheme}_tl.mp4"
        grab(video, 81.300, KEY / f"112739_81.300_{scheme}.jpg")
        video = BASE / "video_20260929_112454" / "scheduled" / f"{scheme}_tl.mp4"
        grab(video, 0.900, KEY / f"112454_00.900_{scheme}.jpg")
    for scheme in SCHEMES:
        video = BASE / "video_20260929_112739" / "upper_bound" / f"{scheme}_tl.mp4"
        grab(video, 81.300, KEY / f"112739_81.300_upper_{scheme}.jpg")

    print("assets written:", len(list(ASSETS.rglob("*"))), "files")


if __name__ == "__main__":
    main()
