"""Profile the actual native cockpit without replacing inference or source frames.

--revision loads the four historical implementation files in memory for a fair
comparison. Model files, thresholds and input video are never changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FILES = {
    'competition_io.hud_simulator': 'competition_io/hud_simulator.py',
    'tools.competition_console.view': 'tools/competition_console/view.py',
    'tools.competition_console.split_video': 'tools/competition_console/split_video.py',
    'tools.run_competition_console': 'tools/run_competition_console.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--headful', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--cv-threads', type=int)
    parser.add_argument('--unpaced', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.cv_threads is not None:
        cv2.setNumThreads(args.cv_threads)
    sources = {}
    if args.revision:
        for name, relative in FILES.items():
            importlib.import_module(name.rsplit('.', 1)[0])
            source = subprocess.check_output(['git', 'show', f'{args.revision}:{relative}'], cwd=ROOT)
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / relative)
            module.__package__ = name.rsplit('.', 1)[0]
            sys.modules[name] = module
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
            sources[relative] = hashlib.sha256(source).hexdigest()
    else:
        for relative in FILES.values():
            sources[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        relative = 'tools/competition_console/runtime.py'
        sources[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    runner = importlib.import_module('tools.run_competition_console')
    from tools.competition_console.split_video import SplitVideo
    from tools.competition_console.navigation_reader import NavigationReader
    timings = defaultdict(list)

    def wrap(obj, name, label):
        original = getattr(obj, name)
        def measured(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                key = label(a) if callable(label) else label
                timings[key].append((time.perf_counter()-start)*1000)
        setattr(obj, name, measured)

    wrap(runner.PcDetector, 'detect', lambda a: 'specialist' if a[0].specialist else 'main')
    wrap(SplitVideo, 'split', 'split')
    wrap(NavigationReader, 'update', 'navigation')
    wrap(runner.NativeView, 'render', 'render')
    wrap(runner.MockScenarioHud, 'submit', 'hud_register_check')
    wrap(runner, 'project_scenario', 'hud_projection')
    wrap(cv2, 'imshow', 'present')
    wrap(cv2, 'waitKey', 'ui_wait')
    capture_type, writer_type = cv2.VideoCapture, cv2.VideoWriter
    class Capture:
        def __init__(self, *a, **kw): self.inner = capture_type(*a, **kw)
        def __getattr__(self, key): return getattr(self.inner, key)
        def read(self): return self.inner.read()
    class Writer:
        def __init__(self, *a, **kw): self.inner = writer_type(*a, **kw)
        def __getattr__(self, key): return getattr(self.inner, key)
        def write(self, *a): return self.inner.write(*a)
    wrap(Capture, 'read', 'decode')
    wrap(Writer, 'write', 'encode')
    cv2.VideoCapture, cv2.VideoWriter = Capture, Writer
    command = [
        '--video', 'D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/video_20260929_112739.mp4',
        '--profile', str(ROOT/'configs/competition_native_navigation_112739.json'),
        '--split-navigation', '--mode', 'navigation', '--mock-hud', '--close-at-end',
        '--window-title', 'ADAS | Performance verification', '--start', str(args.start),
        '--max-frames', str(args.frames), '--output', str(args.output/'run'),
    ]
    if not args.headful: command.append('--headless')
    if args.record: command.append('--record')
    if args.unpaced: command.append('--unpaced')
    options = runner.parser().parse_args(command)
    outer = time.perf_counter()
    report = runner.run(options)
    outer = time.perf_counter()-outer
    stages = {key: {'calls': len(values), 'total_s': sum(values)/1000,
                    'mean_ms': float(np.mean(values)), 'p50_ms': float(np.median(values)),
                    'p95_ms': float(np.percentile(values, 95)), 'max_ms': max(values)}
              for key, values in timings.items()}
    result = {'revision': args.revision or 'working_tree', 'source_sha256': sources,
              'headful': args.headful, 'record': args.record, 'cv_threads': cv2.getNumThreads(),
              'outer_wall_s': outer, 'loop_wall_s': report['wall_seconds'],
              'frames': report['processed_frames'], 'source_duration_s': report['processed_frames']/30,
              'threaded_stages_overlap_do_not_sum_to_wall': not bool(args.revision),
              'stages': stages, 'evidence': 'PC_CPU_ACTUAL_INFERENCE_NO_HARDWARE_CLAIM'}
    (args.output/'timings.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
