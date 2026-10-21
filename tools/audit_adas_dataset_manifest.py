#!/usr/bin/env python3
"""Fail-fast audit for ADAS train/eval manifest declarations.

This validates the declared dataset contract only. It CANNOT determine if
bounding boxes are actually complete or correctly annotated.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CLASSES = (
    "person", "bicycle", "motorcycle", "car", "bus", "truck",
    "red_light", "yellow_light", "green_light",
    "traffic_light_other", "crosswalk",
)
REQUIRED_FIELDS = (
    "sample_id", "dataset", "sequence_id", "split",
    "classes_exhaustive", "license", "source_uri",
)
SPLITS = {"train", "val", "test"}


def audit_rows(rows, required_classes=DEFAULT_CLASSES):
    """Return a stable machine-readable audit report.

    `classes_exhaustive` is a pipe-delimited declaration of categories whose
    instances have ALL been annotated in that particular sample.
    For standard YOLO losses, incomplete images must be repaired or excluded
    before mixing datasets; simply retaining the classification head is unsafe.
    """
    required = set(required_classes)
    errors = []
    warnings = []
    seen = set()
    sequence_splits = defaultdict(set)
    counts = Counter()

    if not required:
        errors.append("required classes list is empty")
    if len(required) != len(tuple(required_classes)):
        errors.append("required classes contain duplicates")

    for line_number, row in enumerate(rows, start=2):
        counts["samples"] += 1
        missing = [key for key in REQUIRED_FIELDS if not str(row.get(key, "")).strip()]
        if missing:
            errors.append(f"row {line_number}: empty required fields: {', '.join(missing)}")
            continue

        sample_id = row["sample_id"].strip()
        dataset = row["dataset"].strip()
        sequence_id = row["sequence_id"].strip()
        split = row["split"].strip().lower()
        source = row["source_uri"].strip()
        license_name = row["license"].strip()
        if sample_id in seen:
            errors.append(f"row {line_number}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)

        if split not in SPLITS:
            errors.append(f"row {line_number}: unknown split {split!r}")
            continue
        counts[split] += 1
        sequence_splits[(dataset, sequence_id)].add(split)

        exhaustive = {c.strip() for c in row["classes_exhaustive"].split("|") if c.strip()}
        absent = sorted(required - exhaustive)
        if absent:
            errors.append(
                f"row {line_number}: {sample_id}: missing exhaustive label coverage "
                f"for {', '.join(absent)}; ordinary YOLO training/evaluation cannot "
                "treat these missing labels as confirmed background"
            )
        extra = sorted(exhaustive - required)
        if extra:
            warnings.append(f"row {line_number}: extra declared classes: {', '.join(extra)}")
        if source.startswith("http") and not license_name:
            errors.append(f"row {line_number}: license information absent")

    if not counts["samples"]:
        errors.append("manifest is empty")
    for (dataset, sequence), splits in sorted(sequence_splits.items()):
        if len(splits) > 1:
            errors.append(
                f"sequence leakage: dataset={dataset!r} sequence={sequence!r} "
                f"appears in splits={','.join(sorted(splits))}"
            )
    warnings.append(
        "Metadata PASS does not prove that actual image annotations are exhaustive; "
        "manually audit images/labels and respect all licenses."
    )
    return {
        "status": "PASS" if not errors else "FAIL",
        "counts": dict(sorted(counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--required-classes", default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class names; defaults to planned 11-class ADAS model",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report output")
    args = parser.parse_args(argv)
    names = tuple(x.strip() for x in args.required_classes.split(",") if x.strip())

    with args.manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(set(REQUIRED_FIELDS) - set(reader.fieldnames or []))
        if missing_columns:
            report = {
                "status": "FAIL", "counts": {}, "warnings": [],
                "errors": ["missing CSV columns: " + ", ".join(missing_columns)],
            }
        else:
            report = audit_rows(reader, names)

    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
