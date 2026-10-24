"""B-05H index the human BDD holdout GT separately from pseudo labels."""
from pathlib import Path
import json
import shutil
import cv2
from collections import Counter

ROOT = Path(r"D:\Datasets\adas_edge_training\b03c_rknn_validation_20261021\board_holdout50")
GT = Path(r"D:\Datasets\adas_edge_training\b03d_gt_hf_20261021")
OUT = Path(r"D:\Datasets\adas_edge_training\b05h_auto_annotation_20261024\real_gt")


def main():
    for sub in ("images/test", "labels/test", "meta"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    colors = Counter()
    for path in sorted(GT.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        image = ROOT / (data["name"] + ".jpg")
        if not image.exists():
            continue
        frame = cv2.imread(str(image))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        label_path = OUT / "labels" / "test" / (data["name"] + ".txt")
        lines = []
        for frame_obj in data.get("frames", []):
            for obj in frame_obj.get("objects", []):
                if obj.get("category") != "traffic light":
                    continue
                box = obj.get("box2d") or {}
                color = str((obj.get("attributes") or {}).get("trafficLightColor", "none")).lower()
                try:
                    x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
                except (KeyError, TypeError, ValueError):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n")
                rows.append({"image": data["name"] + ".jpg", "bbox_xyxy": [x1, y1, x2, y2],
                             "color": color, "source": "bdd_human_gt"})
                colors[color] += 1
        if lines:
            label_path.write_text("".join(lines), encoding="utf-8")
            shutil.copy2(image, OUT / "images" / "test" / image.name)
    (OUT / "dataset.yaml").write_text(
        f"path: {OUT}\ntest: images/test\nnames:\n  0: traffic light\n", encoding="utf-8")
    (OUT / "meta" / "human_gt.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT / "meta" / "summary.json").write_text(json.dumps({
        "frames": len({r["image"] for r in rows}), "boxes": len(rows),
        "colors": dict(colors), "source": "BDD100K human GT",
        "usage": "independent evaluation only; not in pseudo training",
    }, indent=2), encoding="utf-8")
    print("real GT", len(rows), colors)


if __name__ == "__main__":
    main()
