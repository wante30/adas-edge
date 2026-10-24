"""Build B-05I ROI training/domain and fixed BDD50 evaluation sets."""
from pathlib import Path
import json, cv2, shutil

CLEAN = Path(r"D:\Datasets\adas_edge_training\b05i_clean_20261024")
DATA = Path(r"D:\Datasets\adas_edge_training\tl_specialist\data_512x192")
BDD_IMG = Path(r"D:\Datasets\adas_edge_training\b03c_rknn_validation_20261021\board_holdout50")
BDD_GT = Path(r"D:\Datasets\adas_edge_training\b03d_gt_hf_20261021")
OUT = Path(r"D:\Datasets\adas_edge_training\b05i_train_20261024")


def roi_image(im):
    return cv2.resize(im[:432], (512, 192), interpolation=cv2.INTER_AREA)


def main():
    for sub in ("domain/images", "domain/labels", "bdd50/images", "bdd50/labels"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    # Domain: accepted clean pseudo images, converted to the Specialist canvas.
    n = 0
    for img_path in sorted((CLEAN / "pseudo_A" / "images" / "train").glob("*.jpg")):
        split = img_path.parent.name
        label = CLEAN / "pseudo_A" / "labels" / split / f"{img_path.stem}.txt"
        if not label.exists():
            continue
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        cv2.imwrite(str(OUT / "domain" / "images" / img_path.name), roi_image(im))
        lines = []
        for line in label.read_text(encoding="utf-8").splitlines():
            _, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
            # Full-image normalized xywh -> top-60% ROI normalized.
            x1, x2 = (cx - bw / 2) * 1280, (cx + bw / 2) * 1280
            y1, y2 = (cy - bh / 2) * 720, (cy + bh / 2) * 720
            y1, y2 = max(0, min(432, y1)), max(0, min(432, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"0 {(x1+x2)/2/1280:.6f} {(y1+y2)/2/432:.6f} {(x2-x1)/1280:.6f} {(y2-y1)/432:.6f}\n")
        (OUT / "domain" / "labels" / f"{img_path.stem}.txt").write_text("".join(lines), encoding="utf-8")
        n += 1
    # Fixed BDD50 evaluation, human GT only.
    count = 0
    for gt_path in sorted(BDD_GT.glob("*.json")):
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        src = BDD_IMG / (data["name"] + ".jpg")
        if not src.exists():
            continue
        im = cv2.imread(str(src))
        if im is None:
            continue
        cv2.imwrite(str(OUT / "bdd50" / "images" / src.name), roi_image(im))
        lines = []
        for fr in data.get("frames", []):
            for obj in fr.get("objects", []):
                if obj.get("category") != "traffic light":
                    continue
                b = obj["box2d"]
                x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
                x1, x2 = max(0, x1), min(1280, x2)
                y1, y2 = max(0, y1), min(432, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(f"0 {(x1+x2)/2/1280:.6f} {(y1+y2)/2/432:.6f} {(x2-x1)/1280:.6f} {(y2-y1)/432:.6f}\n")
        (OUT / "bdd50" / "labels" / f"{src.stem}.txt").write_text("".join(lines), encoding="utf-8")
        count += 1
    (OUT / "baseline.yaml").write_text(
        f"path: {DATA}\ntrain: images/train\nval: images/val\nnames:\n  0: traffic light\n", encoding="utf-8")
    (OUT / "candidate.yaml").write_text(
        f"path: {DATA}\ntrain:\n  - {DATA / 'images/train'}\n  - {OUT / 'domain/images'}\n"
        f"val: images/val\nnames:\n  0: traffic light\n", encoding="utf-8")
    (OUT / "bdd50.yaml").write_text(
        f"path: {OUT / 'bdd50'}\ntest: images\nnames:\n  0: traffic light\n", encoding="utf-8")
    (OUT / "index.json").write_text(json.dumps({
        "domain_images": n,
        "bdd50_images": count,
        "baseline_yaml": str(OUT / "baseline.yaml"),
        "candidate_yaml": str(OUT / "candidate.yaml"),
        "bdd50_yaml": str(OUT / "bdd50.yaml"),
        "source_checkpoint": str(Path(r"D:\Datasets\adas_edge_training\tl_specialist\tl-512x192\weights\best.pt")),
    }, indent=2), encoding="utf-8")
    print("domain", n, "bdd50", count)


if __name__ == "__main__":
    main()

