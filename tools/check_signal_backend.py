"""Run real-photo backend regressions on ARM or PC without YOLO/pytest."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from competition_ai.signal_shape import ContourTemplateBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=Path('tests/fixtures/real_signals'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    backend = ContourTemplateBackend()
    results = []
    for row in json.loads((args.fixtures/'manifest.json').read_text(encoding='utf-8')):
        if 'expected_color' not in row:
            continue
        path = args.fixtures/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Fixture hash mismatch: '+str(path))
        image = cv2.imread(str(path))
        started = time.perf_counter()
        attr = backend.classify(image)
        results.append(dict(file=row['file'], **asdict(attr),
            elapsed_ms=(time.perf_counter()-started)*1000,
            passed=(attr.color, attr.shape) == (row['expected_color'], row['expected_shape'])))
    report = dict(scope='REGRESSION_CHECK_NOT_ACCURACY', opencv=cv2.__version__,
                  python=sys.version, passed=sum(r['passed'] for r in results), total=len(results), results=results)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in ('scope', 'opencv', 'passed', 'total')}))
    if not all(row['passed'] for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
