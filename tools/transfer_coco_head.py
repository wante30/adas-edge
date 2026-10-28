"""Narrow a pretrained COCO detection head to the seven ADAS classes.

The pilot keeps the 80-class head (option A), so this tool covers the deployment
half of the plan: a seven-class graph for RK3566 that does **not** throw the
pretrained classifier away.

Why not a plain row copy into a fresh seven-class head: in Ultralytics' YOLO11
``Detect`` head the class trunk width is itself derived from ``nc``
(``c3 = max(ch[0], min(nc, 100))``), so an ``nc=7`` head builds a *64*-wide
trunk where the ``nc=80`` head had an *80*-wide one::

    cv3[b] = Sequential(
        Sequential(DWConv(c_in -> c3, 3), Conv(c3 -> c3, 1)),
        Sequential(DWConv(c3 -> c3, 3),     Conv(c3 -> c3, 1)),
        Conv2d(c3 -> nc, 1),
    )

There are no rows to copy for those inner tensors - the layouts disagree.

The transfer therefore keeps the entire pretrained class trunk at width 80 and
narrows only the terminal 1x1 convolution, selecting the COCO output rows for
person 0, bicycle 1, car 2, motorcycle 3, bus 5, truck 7 and traffic light 9.
Every backbone, neck, box-regression and class-trunk tensor is preserved bit for
bit, and the seven surviving logits are provably identical to the COCO ones.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

DEFAULT_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light")
COCO_SOURCES: dict[str, int] = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7, "traffic light": 9,
}
TERMINAL = 2  # index of the terminal 1x1 class convolution inside each cv3 branch


def selected_rows(classes: tuple[str, ...] = DEFAULT_CLASSES) -> list[int]:
    return [COCO_SOURCES[name] for name in classes]


def build_narrow_head(source_model, classes: tuple[str, ...] = DEFAULT_CLASSES):
    """Return a copy of ``source_model`` whose terminal class convs emit ``classes``.

    The class trunk keeps its pretrained width, so its features are unchanged;
    only the final projection is cropped to the selected COCO rows.
    """
    import torch
    import torch.nn as nn

    rows = selected_rows(classes)
    model = copy.deepcopy(source_model)
    head = model.model[-1]
    if head.nc != max(rows) + 1 and head.nc <= max(rows):
        raise ValueError(f"source head has {head.nc} outputs, cannot select row {max(rows)}")

    for branch in range(head.nl):
        terminal = head.cv3[branch][TERMINAL]
        if terminal.weight.shape[0] != head.nc:
            raise ValueError(
                f"cv3.{branch}.{TERMINAL} emits {terminal.weight.shape[0]} values, expected {head.nc}"
            )
        narrowed = nn.Conv2d(terminal.in_channels, len(classes), kernel_size=1)
        with torch.no_grad():
            narrowed.weight.copy_(terminal.weight[rows])
            narrowed.bias.copy_(terminal.bias[rows])
        head.cv3[branch][TERMINAL] = narrowed

    head.nc = len(classes)
    head.no = len(classes) + head.reg_max * 4
    head.names = {index: name for index, name in enumerate(classes)}
    model.names = dict(head.names)
    model.yaml["nc"] = len(classes)
    return model


def verify_narrow_head(
    source_model,
    target_model,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    atol: float = 1e-5,
) -> dict:
    """Prove the narrowed head reproduces exactly the COCO logits it kept.

    The class trunk must produce bit-identical features (``trunk_max_abs_diff``
    is 0), and the seven surviving logits must match the COCO logits up to fp32
    accumulation noise: a 1x1 convolution emitting 7 channels instead of 80
    accumulates in a different order, so exact bit equality is not expected.
    """
    import torch

    rows = selected_rows(classes)
    source_head = source_model.model[-1]
    target_head = target_model.model[-1]
    report: dict = {"branches": {}}
    with torch.no_grad():
        for branch in range(source_head.nl):
            channels = source_head.cv3[branch][0][0].conv.in_channels
            probe = torch.randn(1, channels, 8, 8)
            source_features = source_head.cv3[branch][:-1](probe)
            target_features = target_head.cv3[branch][:-1](probe)
            trunk_difference = float((source_features - target_features).abs().max())
            reference = source_head.cv3[branch][TERMINAL](source_features)
            produced = target_head.cv3[branch][TERMINAL](target_features)
            difference = float((produced - reference[:, rows]).abs().max())
            scale = float(reference[:, rows].abs().max()) or 1.0
            report["branches"][f"cv3.{branch}"] = {
                "trunk_max_abs_diff": trunk_difference,
                "max_abs_diff": difference,
                "max_relative_diff": difference / scale,
                "matches": trunk_difference == 0.0 and difference <= atol,
                "logits": int(produced.shape[1]),
            }
    report["all_match"] = all(entry["matches"] for entry in report["branches"].values())
    report["selected_rows"] = rows
    report["atol"] = atol
    return report


def untouched_tensors(source_model, target_model) -> dict:
    """Confirm everything except the terminal class convs is bit-identical."""
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    narrowed, identical, differing = [], 0, []
    for key, value in target_state.items():
        if key not in source_state:
            differing.append(key)
            continue
        source_value = source_state[key]
        if tuple(source_value.shape) != tuple(value.shape):
            narrowed.append(key)
            continue
        if source_value.dtype.is_floating_point:
            if not bool((source_value == value).all()):
                differing.append(key)
                continue
        elif not bool((source_value == value).all()):
            differing.append(key)
            continue
        identical += 1
    return {"identical_tensors": identical, "narrowed_tensors": narrowed, "unexpected_differences": differing}


def build_checkpoint(weights: Path, out: Path, verify: bool = True) -> dict:
    import torch
    from ultralytics import YOLO

    source_model = YOLO(str(weights)).model
    target = build_narrow_head(source_model, DEFAULT_CLASSES)
    target.eval()

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": target.float()}, out)

    summary: dict = {
        "source_weights": str(weights),
        "output": str(out),
        "classes": list(DEFAULT_CLASSES),
        "coco_rows": selected_rows(DEFAULT_CLASSES),
        "nc": len(DEFAULT_CLASSES),
    }
    if verify:
        summary["state_dict"] = untouched_tensors(source_model, target)
        summary["verification"] = verify_narrow_head(source_model, target)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    args = parser.parse_args()

    summary = build_checkpoint(args.weights, args.out, verify=not args.no_verify)
    text = json.dumps(summary, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)

    if summary.get("state_dict", {}).get("unexpected_differences"):
        raise SystemExit("unexpected weight differences outside the terminal class conv")
    if not summary.get("verification", {}).get("all_match", True):
        raise SystemExit("verification failed: the narrowed head does not reproduce the COCO logits")


if __name__ == "__main__":
    main()
