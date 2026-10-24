"""Extract candidate-only frames for human review (no video GT is claimed).

Selects frames where one model has a trusted (>=0.25) box and the baseline has
none, plus the reverse.  These are review candidates, not confirmed false
positives or misses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

COLOR_BGR = {"RED": (40, 40, 235), "GREEN": (40, 210, 60), "YELLOW": (30, 220, 240), "UNKNOWN": (220, 220, 220)}


def draw(frame: np.ndarray, detections: list[dict[str, Any]], caption: str) -> np.ndarray:
    canvas = frame.copy()
    for item in detections:
        if float(item["confidence"]) < 0.25:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in item["bbox"])
        color = str(item.get("color", "UNKNOWN"))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BGR.get(color, COLOR_BGR["UNKNOWN"]), 2)
        cv2.putText(canvas, f"{color} {item['confidence']:.2f}", (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BGR.get(color, COLOR_BGR["UNKNOWN"]), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (18, 24, 28), -1)
    cv2.putText(canvas, caption, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 245, 248), 2, cv2.LINE_AA)
    return canvas


def contact_sheet(items: list[tuple[np.ndarray, str]], path: Path) -> None:
    if not items:
        return
    width, height = 640, 360
    cols = 2
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * (height + 24), cols * (width + 8), 3), 28, np.uint8)
    for index, (image, caption) in enumerate(items):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, cols)
        y, x = row * (height + 24), col * (width + 8)
        sheet[y:y + height, x:x + width] = image
        cv2.putText(sheet, caption[:110], (x + 4, y + height + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (235, 240, 245), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def select(frames_model: list[dict[str, Any]], frames_base: list[dict[str, Any]], fps: float, limit: int = 6) -> list[dict[str, Any]]:
    candidates = []
    for frame, base in zip(frames_model, frames_base):
        trusted = [item for item in frame.get("detections", []) if float(item["confidence"]) >= 0.25]
        base_trusted = [item for item in base.get("detections", []) if float(item["confidence"]) >= 0.25]
        if trusted and not base_trusted:
            best = max(trusted, key=lambda item: float(item["confidence"]))
            candidates.append({"frame": int(frame["frame"]), "seconds": round(float(frame["frame"]) / fps, 3),
                               "detections": trusted, "best_conf": float(best["confidence"])})
    candidates.sort(key=lambda row: -row["best_conf"])
    selected = []
    for row in candidates:
        if all(abs(row["seconds"] - other["seconds"]) >= 2.0 for other in selected):
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timelines", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--baseline", default="yolo11n")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    videos = sorted({p.stem for p in (args.timelines / args.candidate).glob("*.json")})
    records = {}
    for video in videos:
        base = json.loads((args.timelines / args.baseline / f"{video}.json").read_text(encoding="utf-8"))
        cand = json.loads((args.timelines / args.candidate / f"{video}.json").read_text(encoding="utf-8"))
        fps = float(cand["summary"]["source_fps"])
        capture = cv2.VideoCapture(str(args.videos / f"{video}.mp4"))
        if not capture.isOpened():
            raise FileNotFoundError(args.videos / f"{video}.mp4")
        selected = select(cand["frames"], base["frames"], fps, args.limit)
        reverse = select(base["frames"], cand["frames"], fps, args.limit)
        records[video] = {"candidate_only": selected, "baseline_only": reverse}
        for kind, rows in (("candidate_only", selected), ("baseline_only", reverse)):
            items = []
            for row in rows:
                capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
                ok, frame = capture.read()
                if not ok:
                    continue
                scene = draw(frame, row["detections"], f"{args.candidate if kind == 'candidate_only' else args.baseline} "
                                                    f"{kind} {row['seconds']:.2f}s conf={row['best_conf']:.2f}")
                cv2.imwrite(str(args.output / f"{video}_{kind}_{row['frame']:06d}.jpg"), scene,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                items.append((scene, f"{video} {kind} {row['seconds']:.2f}s {row['best_conf']:.2f}"))
            contact_sheet(items, args.output / f"{video}_{kind}.jpg")
        capture.release()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({video: {kind: len(rows) for kind, rows in value.items()} for video, value in records.items()}, indent=2))


if __name__ == "__main__":
    main()
