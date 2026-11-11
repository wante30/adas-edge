"""Commit only bounded, measured result summaries on the authorized PR branch."""
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

from .dataset_adapter import read_json, file_hash


BRANCH = 'feature/issue15-fpga-road-seg-20261112'


def publish(repo, output, push=False):
    repo, output = Path(repo), Path(output)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, text=True, encoding='utf-8').strip()
    if git('branch', '--show-current') != BRANCH:
        raise ValueError('refusing result publication on an unrelated branch')
    rows = list(csv.DictReader((output/'metrics.csv').open(encoding='utf-8')))
    if {r['model'] for r in rows} != {'candidate', 'candidate-plus', 'twin-small'}:
        raise ValueError('all three measured model rows are required for final publication')
    for row in rows:
        run = read_json(output/'checkpoints'/row['model']/'report.json')
        evaluation = read_json(output/'metrics'/f"{row['model']}-heldout.json")
        if run['evidence'] != 'real_manifest' or evaluation['samples'] != 10000:
            raise ValueError('real training and full official val evaluation are required')
        if evaluation['checkpoint_sha256'] != file_hash(output/'checkpoints'/row['model']/'best.pt'):
            raise ValueError('evaluation/checkpoint hash mismatch')
    if not list((output/'exports').glob('*/export-report.json')):
        raise ValueError('measured quantization report is required')
    target = repo/'experiments/training/fpga_road_seg/reports/real_bdd_final'
    staged = git('diff', '--cached', '--name-only').splitlines()
    prefix = target.relative_to(repo).as_posix() + '/'
    if any(not name.startswith(prefix) for name in staged):
        raise ValueError('unrelated staged changes exist; refusing an automatic commit')
    target.mkdir(parents=True, exist_ok=True)
    names = ('README_RESULT.md', 'metrics.csv', 'model_reports.json', 'training_curves.png')
    copied = []
    for name in names:
        dest = target/name
        shutil.copy2(output/name, dest)
        copied.append(dest)
    for src in sorted((output/'exports').glob('*/export-report.json')):
        dest = target/f'{src.parent.name}-export-report.json'
        shutil.copy2(src, dest); copied.append(dest)
    for src in sorted((output/'comparisons'/'all_models').glob('*.jpg'))[:8]:
        dest = target/'comparisons'/src.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dest); copied.append(dest)
    for src in copied:
        if src.stat().st_size > 5*1024*1024:
            raise ValueError(f'unexpectedly large summary artifact: {src}')
    git('add', '-f', '--', *[str(p.relative_to(repo)) for p in copied])
    if git('diff', '--cached', '--name-only'):
        git('commit', '-m', 'results(issue15): publish measured BDD model and INT8 comparisons')
    if push:
        git('-c', 'http.proxy=http://127.0.0.1:7890', 'push', 'origin', f'HEAD:{BRANCH}')
    return git('rev-parse', 'HEAD')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--push', action='store_true')
    args=parser.parse_args()
    print(publish(Path(__file__).resolve().parents[3], args.output, args.push), flush=True)


if __name__=='__main__':
    main()
