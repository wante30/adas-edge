"""Compare native rendering and same-frame geometry against saved source code.

This does not run inference, record video, or change the input/crop resolution.
Decode, source hashing, equivalence checks and profiling are outside timed loops.
Run only when the machine is reserved for this benchmark.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import pstats
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from competition_ai.scenario import Controls, Decision, Direction, Maneuver, Motion
from tools.review_competition_cockpit import observation

REFERENCE = ROOT / "outputs/cockpit_perf_20261016/render/reference_original"
PROFILE = ROOT / "configs/competition_native_navigation_112739.json"
TRACE = ROOT / "outputs/navigation_112739_20261016/recording/trace.jsonl"
FRAME_IDS = (0, 300, 600, 900, 1200, 1310, 1500, 2010, 2400, 2730)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reference implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # A saved view.py is outside the source tree; its assets remain in ROOT.
    if hasattr(module, "ROOT"):
        module.ROOT = ROOT
    return module


def pixel_check(expected, actual):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {"equal": False, "shape_equal": expected.shape == actual.shape,
                "dtype_equal": expected.dtype == actual.dtype}
    difference = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
    return {"equal": not bool(np.any(difference)),
            "different_pixels": int(np.count_nonzero(np.any(difference, axis=2))),
            "max_channel_error": int(difference.max(initial=0)),
            "shape": list(actual.shape), "dtype": str(actual.dtype)}


def render_equal(reference, candidate, expected, actual):
    pixels = pixel_check(expected, actual)
    buttons_equal = reference.buttons == candidate.buttons
    rect_equal = reference.content_rect == candidate.content_rect
    hits_equal = all(reference.hit((r[0]+r[2])/2, (r[1]+r[3])/2)
                     == candidate.hit((r[0]+r[2])/2, (r[1]+r[3])/2)
                     for _, r in reference.buttons)
    return {"pixels": pixels, "buttons_equal": buttons_equal,
            "content_rect_equal": rect_equal, "button_hits_equal": hits_equal,
            "passed": pixels["equal"] and buttons_equal and rect_equal and hits_equal}


@dataclass
class Case:
    frame_id: int
    source: np.ndarray
    road: np.ndarray
    navigation_image: np.ndarray
    snapshot: object
    controls: Controls
    decision: Decision
    kwargs: dict

    def render(self, view):
        return view.render(self.road, self.snapshot, self.controls, self.decision,
                           navigation_image=self.navigation_image, **self.kwargs)


def read_cases(profile, trace_path, splitter, frame_ids):
    wanted = set(frame_ids)
    rows = {}
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["frame"] in wanted:
                rows[row["frame"]] = row
    if set(rows) != wanted:
        raise ValueError(f"Trace missing selected frames: {sorted(wanted-set(rows))}")
    capture = cv2.VideoCapture(profile["video"])
    if not capture.isOpened():
        raise RuntimeError("Cannot open the exact source video")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    cases = []
    try:
        for frame_id in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, source = capture.read()
            if not ok or abs(capture.get(cv2.CAP_PROP_POS_FRAMES)-frame_id-1) > .1:
                raise RuntimeError(f"Could not decode source frame {frame_id}")
            split = splitter.split(source, frame_id)
            row = rows[frame_id]
            if split.metadata != row["same_frame_split"]:
                raise ValueError(f"Recorded observations use different crop geometry at {frame_id}")
            data = dict(row["decision"])
            for name, cls in (("maneuver", Maneuver), ("direction", Direction),
                              ("simulated_motion", Motion), ("navigation_hint", Direction)):
                data[name] = cls(data[name])
            data["alarms"] = tuple(data["alarms"])
            kwargs = {**row["view"], "frame_id": frame_id, "fps": fps,
                      "total_frames": count, "source_name": profile["video"],
                      "navigation": row["navigation"]}
            if kwargs["bev"]:
                raise ValueError("This benchmark covers the current camera/navigation UI, not BEV")
            cases.append(Case(frame_id, source, split.road, split.navigation,
                              observation(row), Controls(**row["controls"]), Decision(**data), kwargs))
    finally:
        capture.release()
    return cases


def statistics(values):
    return {"samples": len(values), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(np.mean(values))}


def timed_pairs(cases, reference, candidate, operation, warmups, rounds):
    for _ in range(warmups):
        for case in cases:
            operation(case, reference)
            operation(case, candidate)
    timings = {"reference": [], "candidate": []}
    for round_id in range(rounds):
        for case_id, case in enumerate(cases):
            pair = [("reference", reference), ("candidate", candidate)]
            if (round_id + case_id) % 2:
                pair.reverse()
            for label, implementation in pair:
                start = time.perf_counter_ns()
                result = operation(case, implementation)
                timings[label].append((time.perf_counter_ns()-start)/1e6)
                del result
    report = {label: statistics(values) for label, values in timings.items()}
    report["median_speedup"] = report["reference"]["median_ms"] / report["candidate"]["median_ms"]
    report["raw_ms"] = timings
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite results")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--frames", type=int, nargs="+", default=FRAME_IDS)
    parser.add_argument("--threads", type=int, nargs="+", default=(1, 2, 4, 32))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--equivalence-only", action="store_true")
    parser.add_argument("--profile-render", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    if args.rounds < 1 or args.warmups < 0 or any(t < 1 for t in args.threads):
        parser.error("rounds/threads must be positive and warmups nonnegative")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if sha256(profile["video"]) != profile["video_sha256"]:
        parser.error("Source video SHA256 mismatch")
    sources = {"reference_view": args.reference / "view.py",
               "reference_split": args.reference / "split_video.py",
               "candidate_view": ROOT / "tools/competition_console/view.py",
               "candidate_split": ROOT / "tools/competition_console/split_video.py"}
    modules = {key: load_module(path, f"cockpit_benchmark_{key}") for key, path in sources.items()}
    reference_split = modules["reference_split"].SplitVideo(profile)
    candidate_split = modules["candidate_split"].SplitVideo(profile)
    reference_view = modules["reference_view"].NativeView()
    candidate_view = modules["candidate_view"].NativeView()
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        cases = read_cases(profile, args.trace, reference_split, args.frames)
        source_hashes = [hashlib.sha256(c.source.tobytes()).hexdigest() for c in cases]
        args.output.mkdir(parents=True, exist_ok=False)
        equivalence = []
        for case in cases:
            old = reference_split.split(case.source, case.frame_id)
            new = candidate_split.split(case.source, case.frame_id)
            old_pixels, new_pixels = case.render(reference_view), case.render(candidate_view)
            check = {"frame_id": case.frame_id, "road": pixel_check(old.road, new.road),
                     "navigation": pixel_check(old.navigation, new.navigation),
                     "metadata_equal": old.metadata == new.metadata,
                     "render": render_equal(reference_view, candidate_view, old_pixels, new_pixels)}
            check["passed"] = all((check["road"]["equal"], check["navigation"]["equal"],
                                   check["metadata_equal"], check["render"]["passed"]))
            equivalence.append(check)
            if not check["passed"]:
                cv2.imwrite(str(args.output/f"mismatch_reference_{case.frame_id}.png"), old_pixels)
                cv2.imwrite(str(args.output/f"mismatch_candidate_{case.frame_id}.png"), new_pixels)
        if not all(row["passed"] for row in equivalence):
            (args.output/"equivalence_failed.json").write_text(json.dumps(equivalence, indent=2), encoding="utf-8")
            raise AssertionError("Pixel/metadata equivalence failed; benchmark rejected")
        reports = []
        if not args.equivalence_only:
            for threads in args.threads:
                cv2.setNumThreads(threads)
                reports.append({"opencv_threads": cv2.getNumThreads(),
                    "split": timed_pairs(cases, reference_split, candidate_split,
                        lambda case, impl: impl.split(case.source, case.frame_id), args.warmups, args.rounds),
                    "render": timed_pairs(cases, reference_view, candidate_view,
                        lambda case, impl: case.render(impl), args.warmups, args.rounds)})
        if args.profile_render:
            cv2.setNumThreads(args.threads[0])
            profiler = cProfile.Profile()
            profiler.enable()
            for case in cases:
                case.render(candidate_view)
            profiler.disable()
            buffer = io.StringIO()
            pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(35)
            (args.output/"render_profile.txt").write_text(buffer.getvalue(), encoding="utf-8")
        unchanged = source_hashes == [hashlib.sha256(c.source.tobytes()).hexdigest() for c in cases]
        report = {"evidence": "PC_STAGE_ONLY_NO_INFERENCE_OR_BOARD_BENCHMARK",
                  "source_video": profile["video"], "source_sha256": profile["video_sha256"],
                  "profile_sha256": sha256(args.profile), "trace_sha256": sha256(args.trace),
                  "code_sha256": {key: sha256(path) for key, path in sources.items()},
                  "environment": {"python": sys.version, "platform": platform.platform(),
                                  "opencv": cv2.__version__, "previous_opencv_threads": previous_threads},
                  "warmup_passes": args.warmups, "rounds": args.rounds,
                  "timing_order": "Alternating reference/candidate per case and round",
                  "timing_excludes": ["decode", "inference", "OCR", "video encode", "source hash", "equivalence", "profiling"],
                  "source_pixels_unchanged": unchanged, "equivalence": equivalence, "benchmarks": reports}
        (args.output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        assert unchanged
        print(json.dumps({"equivalence_passed": len(equivalence), "source_unchanged": unchanged,
                          "measurements": [{"opencv_threads": r["opencv_threads"],
                            **{stage: {k:v for k,v in r[stage].items() if k != "raw_ms"}
                               for stage in ("split", "render")}} for r in reports]}, indent=2))
    finally:
        cv2.setNumThreads(previous_threads)


if __name__ == "__main__":
    main()
