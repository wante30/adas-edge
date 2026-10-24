"""Split B-05H pseudo labels into trusted A, pending B, and negatives C.

The main dataset.yaml deliberately references only A. B stays in pending/ and
must not be silently used as trusted training data.
"""
from pathlib import Path
import json
import shutil
import cv2
import numpy as np

ROOT = Path(r"D:\Datasets\adas_edge_training\b05h_auto_annotation_20261024")
P = ROOT / "pseudo"


def main():
    rows = json.loads((ROOT / "meta" / "pseudo_labels.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        (P / "pending" / "images" / split).mkdir(parents=True, exist_ok=True)
        (P / "pending" / "labels" / split).mkdir(parents=True, exist_ok=True)
    by_image = {}
    for r in rows:
        by_image.setdefault(r["image"], []).append(r)
    a_labels = b_labels = 0
    for image, items in by_image.items():
        if not items:
            continue
        video = items[0]["video"]
        split = "test" if video.startswith("video_20260929_113421") else (
            "val" if video.startswith("video_20260929_112739") else "train")
        img = P / "images" / split / image
        if not img.exists():
            continue
        main_label = P / "labels" / split / f"{Path(image).stem}.txt"
        pending_label = P / "pending" / "labels" / split / f"{Path(image).stem}.txt"
        a_lines, b_lines = [], []
        for r in items:
            x1, y1, x2, y2 = r["bbox_xyxy"]
            w, h = 1280, 720
            line = f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
            if r["review"] == "A":
                a_lines.append(line)
            elif r["review"] == "B":
                b_lines.append(line)
        main_label.write_text("".join(a_lines), encoding="utf-8")
        if b_lines:
            pending_label.write_text("".join(b_lines), encoding="utf-8")
            shutil.copy2(img, P / "pending" / "images" / split / image)
        a_labels += len(a_lines)
        b_labels += len(b_lines)
    pending_yaml = (
        f"path: {P / 'pending'}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: traffic light\n")
    (P / "pending" / "dataset.yaml").write_text(pending_yaml, encoding="utf-8")
    (P / "dataset.yaml").write_text(
        f"path: {P}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: traffic light\n",
        encoding="utf-8")
    # Audit bounds/duplicates and build a representative overlay sheet.
    bad = 0
    dup = 0
    A = []
    B = []
    for r in rows:
        x1, y1, x2, y2 = r["bbox_xyxy"]
        if x1 < 0 or y1 < 0 or x2 > 1280 or y2 > 720 or x2 <= x1 or y2 <= y1:
            bad += 1
        if r["review"] == "A":
            A.append(r)
        elif r["review"] == "B":
            B.append(r)
    # Duplicate boxes per image.
    for image, items in by_image.items():
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                ax1, ay1, ax2, ay2 = a["bbox_xyxy"]
                bx1, by1, bx2, by2 = b["bbox_xyxy"]
                inter = max(0., min(ax2, bx2) - max(ax1, bx1)) * max(0., min(ay2, by2) - max(ay1, by1))
                aa = (ax2 - ax1) * (ay2 - ay1)
                bb = (bx2 - bx1) * (by2 - by1)
                if aa + bb - inter > 0 and inter / (aa + bb - inter) > .85:
                    dup += 1
    summary = {
        "frames": 62, "labels_total": len(rows), "A": len(A), "B": len(B), "C": 0,
        "A_labels_written": a_labels, "B_labels_pending": b_labels,
        "bad_boxes": bad, "near_duplicate_pairs": dup,
        "main_dataset": str(P), "pending_dataset": str(P / "pending"),
        "human_gt": str(ROOT / "real_gt"), "status": "A training-ready only; B pending; no C fabricated",
    }
    (ROOT / "meta" / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def sheet(items, path, title):
        canvas = np.full((720 * 3, 1280 * 2, 3), 30, np.uint8)
        for idx, r in enumerate(items[:6]):
            img = cv2.imread(str(P / "images" / ("test" if r["video"].startswith("video_20260929_113421") else "val" if r["video"].startswith("video_20260929_112739") else "train") / r["image"]))
            if img is None:
                continue
            x1, y1, x2, y2 = map(int, r["bbox_xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, f'{r["review"]} {r["color"]} {r["confidence"]:.2f}',
                        (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
            row, col = divmod(idx, 2)
            canvas[row * 720:(row + 1) * 720, col * 1280:(col + 1) * 1280] = img
        cv2.imwrite(str(path), canvas)
    sheet(A, ROOT / "meta" / "contact_A.jpg", "A")
    sheet(B, ROOT / "meta" / "contact_B.jpg", "B")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
