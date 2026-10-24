"""Plot raw BDD50 PR curves from full B-05L evaluation JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05l_eval import load_gt, ranked_matching  # noqa: E402

COLORS = {"yolo11n": "#1f77b4", "v8n": "#2ca02c", "v8n_p2": "#d62728"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True, help="LABEL=full_eval.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for value in args.eval:
        label, path = value.split("=", 1)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        images = Path(data["dataset"]["images_dir"])
        gt_dir = Path(data["dataset"]["gt_dir"])
        _, gt, counts = load_gt(images, gt_dir, float(data["dataset"]["roi_top_fraction"]))
        rows = ranked_matching(gt, data["predictions_by_image"], floor=float(data["thresholds"]["ap_floor"]),
                               iou_threshold=float(data["thresholds"]["match_iou"]))
        tp = np.cumsum([row["tp"] for row in rows], dtype=np.float64)
        fp = np.cumsum([1 - row["tp"] for row in rows], dtype=np.float64)
        recall = tp / counts["roi_gt"]
        precision = tp / np.maximum(tp + fp, 1.0)
        ax.step(recall, precision, where="post", label=f"{label} AP50={data['metrics_roi']['ap50']:.4f}",
                color=COLORS.get(label), linewidth=2)
    ax.set_xlabel("Recall (150 ROI GT)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title("BDD50 traffic-light PR @ IoU 0.5")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
