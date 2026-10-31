"""Build RKNN calibration sets from the BDD train split."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path(r"D:\Datasets\adas_edge_training\b05n_bdd_multiclass\images\train"))
    ap.add_argument("--out-root", type=Path, default=Path(r"D:\Datasets\adas_edge_training\rknn_b05o"))
    ap.add_argument("--counts", type=int, nargs="+", default=[200, 500, 1000])
    ap.add_argument("--roi-top", type=float, default=0.60)
    args = ap.parse_args()
    names = sorted(args.images.glob("*.jpg"))
    print("train images", len(names))
    for count in args.counts:
        full_dir = args.out_root / f"calib_train_full_{count}"
        spec_dir = args.out_root / f"calib_train_spec_{count}"
        full_dir.mkdir(parents=True, exist_ok=True)
        spec_dir.mkdir(parents=True, exist_ok=True)
        full_list, spec_list = [], []
        for index, path in enumerate(names[:count]):
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            height = bgr.shape[0]
            rgb_full = cv2.cvtColor(cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            out = full_dir / f"{index:04d}.jpg"
            cv2.imwrite(str(out), cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            full_list.append(str(out))
            roi = bgr[: int(round(height * args.roi_top))]
            rgb_spec = cv2.cvtColor(cv2.resize(roi, (512, 192), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            out = spec_dir / f"{index:04d}.jpg"
            cv2.imwrite(str(out), cv2.cvtColor(rgb_spec, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            spec_list.append(str(out))
        (args.out_root / f"calib_train_full_{count}.txt").write_text("\n".join(full_list) + "\n", encoding="utf-8")
        (args.out_root / f"calib_train_spec_{count}.txt").write_text("\n".join(spec_list) + "\n", encoding="utf-8")
        print("written", count, len(full_list), len(spec_list))


if __name__ == "__main__":
    main()
