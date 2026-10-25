"""Build RKNN calibration lists (full-frame 512x288 and specialist 512x192)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

FULL = Path(r"D:\Datasets\adas_edge_training\rknn_b05o\calib_full")
SPEC = Path(r"D:\Datasets\adas_edge_training\rknn_b05o\calib_spec")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path(r"D:\Datasets\adas_edge_training\b05n_bdd_multiclass\images\test"))
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--roi-top", type=float, default=0.60)
    args = ap.parse_args()
    FULL.mkdir(parents=True, exist_ok=True)
    SPEC.mkdir(parents=True, exist_ok=True)
    names = sorted(args.images.glob("*.jpg"))[: args.count]
    full_list, spec_list = [], []
    for index, path in enumerate(names):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        height, width = bgr.shape[:2]
        rgb_full = cv2.cvtColor(cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        out = FULL / f"{index:04d}.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        full_list.append(str(out))
        roi = bgr[: int(round(height * args.roi_top))]
        rgb_spec = cv2.cvtColor(cv2.resize(roi, (512, 192), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        out = SPEC / f"{index:04d}.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(rgb_spec, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        spec_list.append(str(out))
    (FULL.parent / "calib_full.txt").write_text("\n".join(full_list) + "\n", encoding="utf-8")
    (FULL.parent / "calib_spec.txt").write_text("\n".join(spec_list) + "\n", encoding="utf-8")
    print("written", len(full_list), len(spec_list))


if __name__ == "__main__":
    main()
