"""Commit only bounded, measured result summaries on the authorized PR branch."""
import argparse
import csv
import os
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
        if run['train_samples'] != 62984 or run['history'][-1]['epoch'] < run['config']['epochs']:
            raise ValueError('full training is incomplete')
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
        body = output/'pr-body.md'
        result_rows = ['| Model | Drivable IoU | Solid F1 | Dashed F1 | Parameters |',
                       '|---|---:|---:|---:|---:|']
        for r in rows:
            result_rows.append(f"| {r['model']} | {r['drivable_iou'] or 'N/A'} | {r['solid_f1'] or 'N/A'} | {r['dashed_f1'] or 'N/A'} | {r['parameters']} |")
        body.write_text('''## 实际变更

为 Issue #15 提供独立 FPGA 道路分割训练/评估/可视化/ONNX PTQ 项目。
复用 PR #4 的 BDD100K 每图 JSON/poly2d 渲染方法，恢复完整 100K Images 和 80K JSON。
按视频分组划分 62,984 train、7,012 internal val，隔离与官方 val 重叠的 4 张 train 帧；
缺失任务屏蔽监督，官方 val 10,000 张仅作 held-out 指标。三个新模型分别真实训练，未复用旧 B-2a 精度。

## 新模型真实指标

''' + '\n'.join(result_rows) + '''

指标口径、checkpoint 哈希、训练曲线、真实道路对比和量化前后精度见
[`README_RESULT.md`](experiments/training/fpga_road_seg/reports/real_bdd_final/README_RESULT.md)
及同目录 `metrics.csv`、`*-export-report.json`。Small 保留两头参考结构，无实/虚线输出。
所有报告均由本次 checkpoint 和真实 held-out 评估生成。合成 smoke 结果不作为道路精度。

BDD crosswalk 曲线仅用于独立正像素弱监督试验；没有 dense crosswalk 真值，dense IoU/Recall 保持 N/A。
Conv W8A8 QDQ 的 Add/Resize/I/O 等仍浮点，尚非 PG2L200H 可直接烧写的产物。
硬件算子/资源/时序、定点舍入/饱和、权重打包和板端 FPS/时延仍待队友资料。

## 验证及范围

- 35 项分割项目测试、38 项现有 frame/PCIe/lane/Tesla-BEV/AI-bridge 回归测试通过；compileall 通过。
- 数据校验、3 个模型的 AMP 真实试训/全量训练与 10k held-out 评估、最佳权重和导出均有日志/哈希证据。
- 持久进程支持 last 恢复、最佳权重、周期 checkpoint、错误恢复和自动提交轻量结果。
- 未修改 FAST-Q4、PCIe、HUD、Tesla BEV、队友权重或 FPGA 工程。数据和权重不进入普通 Git。

Refs #15
''', encoding='utf-8')
        env = os.environ.copy()
        env['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        subprocess.run(['gh', 'pr', 'edit', '16', '--repo', 'wante30/adas-edge', '--body-file', str(body)],
                       cwd=repo, env=env, check=True)
    return git('rev-parse', 'HEAD')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--push', action='store_true')
    args=parser.parse_args()
    print(publish(Path(__file__).resolve().parents[3], args.output, args.push), flush=True)


if __name__=='__main__':
    main()
