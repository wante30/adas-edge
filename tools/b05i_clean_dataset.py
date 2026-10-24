"""B-05I clean dataset: reviewed B accepts, no pending/empty negative leakage.

The 43 B candidates were inspected as crop contact sheets by the image-capable
agent.  Only the explicit accepted indices below enter A.  Rejected/uncertain
items remain in metadata, not as empty labels.
"""
from pathlib import Path
import json, shutil, cv2

ROOT = Path(r"D:\Datasets\adas_edge_training\b05h_auto_annotation_20261024")
P = ROOT / "pseudo"
OUT = Path(r"D:\Datasets\adas_edge_training\b05i_clean_20261024")
B_ACCEPT = {0,1,2,3,4,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,28,29,31,35,37}
B_REJECT = set(range(43)) - B_ACCEPT


def split(video):
    return "test" if video.startswith("video_20260929_113421") else (
        "val" if video.startswith("video_20260929_112739") else "train")


def main():
    rows = json.loads((ROOT / "meta" / "pseudo_labels.json").read_text(encoding="utf-8"))
    B_rows = [r for r in rows if r["review"] == "B"]
    A_rows = [r for r in rows if r["review"] == "A"]
    for r, idx in zip(B_rows, range(len(B_rows))):
        r["review"] = "A" if idx in B_ACCEPT else ("REJECT" if idx in B_REJECT else "UNCERTAIN")
        r["reviewer"] = "agent_image_review"
        r["review_reason"] = "visible lamp" if idx in B_ACCEPT else "not a confirmed traffic lamp"
    clean = A_rows + [r for r in B_rows if r["review"] == "A"]
    by_image = {}
    for r in clean:
        by_image.setdefault(r["image"], []).append(r)
    for sub in ("images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"):
        (OUT / "pseudo_A" / sub).mkdir(parents=True, exist_ok=True)
    (OUT / "meta").mkdir(parents=True, exist_ok=True)
    written = []
    for image, items in by_image.items():
        video = items[0]["video"]
        s = split(video)
        src = P / "images" / s / image
        if not src.exists():
            continue
        shutil.copy2(src, OUT / "pseudo_A" / "images" / s / image)
        lines = []
        for r in items:
            x1, y1, x2, y2 = r["bbox_xyxy"]
            lines.append(f"0 {(x1+x2)/2/1280:.6f} {(y1+y2)/2/720:.6f} {(x2-x1)/1280:.6f} {(y2-y1)/720:.6f}\n")
        (OUT / "pseudo_A" / "labels" / s / f"{Path(image).stem}.txt").write_text("".join(lines), encoding="utf-8")
        written.append({"image": image, "split": s, "count": len(items),
                        "colors": [r["color"] for r in items], "source": "agent_reviewed_pseudo"})
    (OUT / "pseudo_A" / "dataset.yaml").write_text(
        f"path: {OUT / 'pseudo_A'}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: traffic light\n",
        encoding="utf-8")
    (OUT / "meta" / "review.json").write_text(json.dumps({
        "B_total": len(B_rows), "B_accept": len(B_ACCEPT), "B_reject": len(B_REJECT),
        "B_accept_indices": sorted(B_ACCEPT), "B_reject_indices": sorted(B_REJECT),
        "A_before": len(A_rows), "A_after": len(clean),
        "empty_negative_images_used": False,
        "reviewer": "agent_image_review", "note": "rejected items are not used as empty labels",
    }, indent=2), encoding="utf-8")
    (OUT / "meta" / "index.json").write_text(json.dumps({"images": written, "A_boxes": len(clean)}, indent=2), encoding="utf-8")
    print("A boxes", len(clean), "images", len(written), "B accepted", len(B_ACCEPT))


if __name__ == "__main__":
    main()
