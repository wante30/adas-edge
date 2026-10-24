"""B-05L plots: PR curves, confidence-recall, size recall and video timelines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"yolo11n": "#1f77b4", "v8n": "#2ca02c", "v8n_p2": "#d62728"}


def load(value: str) -> tuple[str, dict]:
    label, path = value.split("=", 1)
    return label, json.loads(Path(path).read_text(encoding="utf-8"))


def pr_plot(models: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, data in models.items():
        curve = data["curves"]["pr50"]
        ax.plot(curve["recall"], curve["precision"], label=f"{label} AP50={data['metrics_roi']['ap50']:.4f}",
                color=COLORS.get(label), linewidth=2)
    ax.set_xlabel("Recall (150 ROI GT)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title("BDD50 traffic-light PR @ IoU 0.5")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def conf_plot(models: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, data in models.items():
        points = data["curves"]["confidence"]["thresholds"]
        conf = [point["conf"] for point in points]
        recall = [point["recall"] for point in points]
        fppi = [point["fp_per_image"] for point in points]
        axes[0].plot(conf, recall, label=label, color=COLORS.get(label), linewidth=2)
        axes[1].plot(conf, fppi, label=label, color=COLORS.get(label), linewidth=2)
    axes[0].set_xlabel("Confidence threshold")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Confidence-Recall")
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("FP per image")
    axes[1].set_title("Confidence-FP/image")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def size_plot(models: dict, path: Path) -> None:
    groups = ["<8", "8-16", "16-32", ">32"]
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (label, data) in enumerate(models.items()):
        recalls = [data["metrics_roi"]["size_groups"][group]["tp"] / data["metrics_roi"]["size_groups"][group]["gt"] for group in groups]
        ax.bar(x + (index - (len(models) - 1) / 2) * width, recalls, width, label=label, color=COLORS.get(label))
        for position, value in zip(x + (index - (len(models) - 1) / 2) * width, recalls):
            ax.text(position, value + 0.01, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel("GT width in original 1280x720 pixels")
    ax.set_ylabel("Recall @ conf 0.25")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("BDD50 size-group recall")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def timeline_plot(comparison: dict, path: Path) -> None:
    videos = list(comparison["videos"])
    fig, axes = plt.subplots(len(videos), 1, figsize=(12, 2.2 * len(videos)), squeeze=False)
    labels = list(comparison["labels"])
    for row, video in enumerate(videos):
        ax = axes[row][0]
        for index, label in enumerate(labels):
            model = comparison["videos"][video]["models"].get(label)
            if not model:
                continue
            for run in model["trusted_runs"]:
                ax.plot([run["start_s"], run["end_s"]], [index, index], color=COLORS.get(label), linewidth=8,
                        solid_capstyle="butt")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlim(0, comparison["videos"][video]["duration_s"])
        ax.set_xlabel("Video time (s)")
        ax.set_title(f"{video}  trusted detections conf>=0.25")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True, help="LABEL=eval.json")
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    models = dict(load(value) for value in args.eval)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pr_plot(models, args.output_dir / "BDD50_PR50.png")
    conf_plot(models, args.output_dir / "BDD50_confidence_recall.png")
    size_plot(models, args.output_dir / "BDD50_size_recall.png")
    if args.comparison:
        timeline_plot(json.loads(args.comparison.read_text(encoding="utf-8")), args.output_dir / "video_trusted_timeline.png")
    print(json.dumps({"output": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
