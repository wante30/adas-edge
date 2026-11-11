"""Persistent BDD100K download, mask generation, training, evaluation, and export."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODELS = {
    'candidate': 'experiments/training/fpga_road_seg/configs/candidate.json',
    'candidate-plus': 'experiments/training/fpga_road_seg/configs/candidate_plus.json',
    'twin-small': 'experiments/training/fpga_road_seg/configs/twin_small.json',
}


class Pipeline:
    def __init__(self, repo, data, out, epochs=12):
        self.repo = Path(repo).resolve()
        self.data = Path(data).resolve()
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.log_path = self.out / 'run.log'
        self.log = open(self.log_path, 'a', encoding='utf-8', buffering=1)
        self.env = os.environ.copy()
        self.env['PYTHONUNBUFFERED'] = '1'
        self.env['BDD_SEG_CACHE'] = str(self.data / 'tensor_cache_320x192_v1')
        self.env['OMP_NUM_THREADS'] = '1'
        os.environ['BDD_SEG_CACHE'] = self.env['BDD_SEG_CACHE']

    def note(self, message):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
        print(line, file=self.log, flush=True)

    def run(self, args, check=True):
        args = [str(x) for x in args]
        self.note('RUN ' + subprocess.list2cmdline(args))
        result = subprocess.run(args, cwd=self.repo, env=self.env, stdout=self.log,
                                stderr=subprocess.STDOUT, check=False)
        self.note(f'EXIT {result.returncode}')
        if check and result.returncode:
            raise subprocess.CalledProcessError(result.returncode, args)
        return result.returncode

    def wait_archives(self):
        archives = self.data / 'archives'
        images = archives / 'bdd100k_images.zip'
        labels = archives / 'bdd100k_labels.zip'
        while True:
            manifest = archives / 'download_manifest.json'
            if images.is_file() and labels.is_file() and manifest.is_file():
                data = json.loads(manifest.read_text(encoding='utf-8'))
                if set(data) == {images.name, labels.name}:
                    self.note(f'archives_verified images={images.stat().st_size} labels={labels.stat().st_size}')
                    return images, labels
            self.note('starting_resumable_public_archive_download')
            self.run([sys.executable, '-m', 'experiments.training.fpga_road_seg.download_official_bdd',
                      '--output', archives])

    def record_failure(self, name, error):
        folder = self.out / 'failures'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{name}.txt'
        path.write_text(f'{type(error).__name__}: {error}\n', encoding='utf-8')
        self.note(f'FAILED {name}: {type(error).__name__}: {error}')

    def prepare(self, images, labels):
        index = self.out / 'data' / 'bdd_index.jsonl'
        index.parent.mkdir(parents=True, exist_ok=True)
        if not index.exists():
            self.run([sys.executable, '-m', 'experiments.training.fpga_road_seg.prepare_official_bdd',
                      '--root', self.data, '--images-zip', images, '--labels-zip', labels,
                      '--output', index])
        else:
            self.note('using_existing_extracted_bdd_index')
        cache = self.data / 'rendered_masks_w10_v2'
        manifest = cache / 'manifest.jsonl'
        if not manifest.is_file():
            self.run([sys.executable, '-m', 'experiments.training.fpga_road_seg.dataset_adapter',
                      '--index', index, '--output', cache, '--line-width', '10', '--workers', '12', '--resume'])
        self.run([sys.executable, '-m', 'experiments.training.fpga_road_seg.audit_legacy_supervision',
                  '--manifest', manifest])
        return index, manifest

    def train_model(self, name, config, manifest, pilot=False):
        output = self.out / 'checkpoints' / name
        if pilot:
            output = output / 'pilot'
        output.mkdir(parents=True, exist_ok=True)
        common = [sys.executable, '-m', 'experiments.training.fpga_road_seg.train',
                  '--manifest', manifest, '--config', self.repo / config, '--output', output,
                  '--device', 'cuda', '--workers', '8', '--batch-size', '16', '--amp', '--exclude-channel', 'crosswalk']
        last = output / 'last.pt'
        target_epochs = 3 if pilot else self.epochs + 3
        for attempt in range(3):
            report_path = output / 'report.json'
            history = json.loads(report_path.read_text(encoding='utf-8')).get('history', []) if report_path.exists() else []
            if history and history[-1]['epoch'] >= target_epochs:
                break
            args = common + ['--epochs', str(target_epochs)]
            if pilot:
                args += ['--train-limit', '512', '--val-limit', '128']
            resume = last if last.exists() else output / 'pilot' / 'last.pt'
            if resume.exists():
                args += ['--resume', resume]
            if self.run(args, check=False) == 0:
                break
            self.note(f'training_retry model={name} pilot={pilot} attempt={attempt + 1}/3')
        else:
            raise RuntimeError(f'{name} failed after 3 resumable training attempts')
        if not (output / 'best.pt').exists():
            raise FileNotFoundError(f'best checkpoint missing: {output / "best.pt"}')
        if pilot:
            return output
        self.run([sys.executable, '-m', 'experiments.training.fpga_road_seg.eval',
                  '--checkpoint', output / 'best.pt', '--manifest', manifest, '--split', 'test',
                  '--output', self.out / 'metrics' / f'{name}-heldout.json', '--device', 'cuda'])
        self.make_comparisons_and_failures(name, output / 'best.pt', manifest)
        return output

    def make_comparisons_and_failures(self, name, checkpoint, manifest, scan_limit=512):
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from experiments.training.fpga_road_seg.dataset_adapter import RoadDataset, letterbox, restore
        from experiments.training.fpga_road_seg.infer import overlay
        from experiments.training.fpga_road_seg.train import load_checkpoint, names_for

        model, ck = load_checkpoint(checkpoint, 'cuda')
        model.eval()
        cfg = ck['config']
        dataset = RoadDataset(manifest, 'test', (cfg['width'], cfg['height']), name == 'twin-small')
        compare_dir = self.out / 'comparisons' / name
        fail_dir = self.out / 'failures' / name
        compare_dir.mkdir(parents=True, exist_ok=True)
        fail_dir.mkdir(parents=True, exist_ok=True)
        compare_ids = set(np.linspace(0, len(dataset) - 1, min(8, len(dataset)), dtype=int).tolist())
        scored = []
        for i in sorted(set(range(min(scan_limit, len(dataset)))) | compare_ids):
            row = dataset.rows[i]
            with Image.open(row['image']) as im:
                image = np.asarray(im.convert('RGB'))
            inp, geometry = letterbox(image, size=(cfg['width'], cfg['height']))
            with torch.no_grad():
                probability = model(inp[None].cuda()).sigmoid()[0].cpu().numpy()
            native = restore(probability, geometry)
            if 'crosswalk' in ck.get('excluded_channels', ()):
                native[3] = 0
            with np.load(row['masks'], allow_pickle=False) as masks:
                target = masks['target'][0].astype(bool)
                valid = masks['valid'][0].astype(bool)
            prediction = native[0] >= cfg.get('threshold', .5)
            tp = int((prediction & target & valid).sum())
            fp = int((prediction & ~target & valid).sum())
            fn = int((~prediction & target & valid).sum())
            iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
            if i < scan_limit:
                scored.append((iou, i, row['id'], tp, fp, fn))
            if i in compare_ids:
                output = overlay(image, native, names_for(name), f'{name} / held-out BDD val', cfg.get('threshold', .5))
                output.save(compare_dir / f'{row["id"]}.png')

        worst = sorted(scored)[:10]
        for iou, i, sample_id, tp, fp, fn in worst:
            row = dataset.rows[i]
            with Image.open(row['image']) as im:
                original = im.convert('RGB')
                image = np.asarray(original)
            inp, geometry = letterbox(image, size=(cfg['width'], cfg['height']))
            with torch.no_grad():
                probability = model(inp[None].cuda()).sigmoid()[0].cpu().numpy()
            pred = restore(probability, geometry)[0] >= cfg.get('threshold', .5)
            with np.load(row['masks'], allow_pickle=False) as masks:
                target = masks['target'][0].astype(bool)
                valid = masks['valid'][0].astype(bool)
            false_positive = pred & ~target & valid
            false_negative = ~pred & target & valid
            size = (640, 360)
            base = original.resize(size, Image.Resampling.BILINEAR)
            def panel(mask, color):
                resized = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 0
                pixels = np.asarray(base).astype(np.float32)
                pixels[resized] = pixels[resized] * .55 + np.asarray(color, np.float32) * .45
                return Image.fromarray(pixels.clip(0, 255).astype(np.uint8))
            panels = [base.copy(), panel(target, (0, 210, 100)), panel(pred, (0, 210, 100)),
                      panel(false_positive, (255, 65, 55)), panel(false_negative, (60, 120, 255))]
            canvas = Image.new('RGB', (size[0] * len(panels), size[1] + 44), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 3), f'{sample_id} | drivable IoU={iou:.4f} | scan subset {min(scan_limit, len(dataset))}', fill='black')
            for j, (panel_image, title) in enumerate(zip(panels, ('input', 'ground truth', 'prediction', 'false positive', 'false negative'))):
                canvas.paste(panel_image, (j * size[0], 44))
                draw.text((j * size[0] + 4, 26), title, fill='black')
            canvas.save(fail_dir / f'{sample_id}.png')
            (fail_dir / f'{sample_id}.json').write_text(json.dumps(
                dict(sample=sample_id, heldout_split='official BDD100K val',
                     drivable_iou=iou, tp=tp, fp=fp, fn=fn,
                     scan_subset=min(scan_limit, len(dataset)),
                     crosswalk_note='BDD sparse positive curves are not dense labels'),
                indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    def weak_crosswalk_experiment(self, name, config, manifest, source):
        if name == 'twin-small':
            self.note('crosswalk_weak_experiment=not_applicable (reference head has no crosswalk output)')
            return
        output = self.out / 'checkpoints' / f'{name}-weak-crosswalk'
        output.mkdir(parents=True, exist_ok=True)
        last = output / 'last.pt'
        if last.exists():
            report = output / 'report.json'
            if report.exists() and len(json.loads(report.read_text(encoding='utf-8')).get('history', [])) >= self.epochs + 6:
                return
        self.run([sys.executable, '-m', 'experiments.training.fpga_road_seg.train',
                  '--manifest', manifest, '--config', self.repo / config, '--output', output,
                  '--device', 'cuda', '--workers', '8', '--batch-size', '16', '--amp', '--epochs', str(self.epochs + 6),
                  '--resume', last if last.exists() else source / 'last.pt'])
        self.note(f'{name}: weak crosswalk fine-tune is positive-curve-only; do not interpret as dense IoU')

    def plot_curves(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
        found = False
        for name in MODELS:
            report = self.out / 'checkpoints' / name / 'report.json'
            if not report.exists():
                continue
            hist = json.loads(report.read_text(encoding='utf-8'))['history']
            ax.plot([e['epoch'] for e in hist], [e['train_loss'] for e in hist], label=f'{name} train')
            ax.plot([e['epoch'] for e in hist], [e['validation']['loss'] for e in hist], '--', label=f'{name} val')
            found = True
        if found:
            ax.set(xlabel='Epoch', ylabel='Masked BCE + Dice loss', title='BDD100K real-data training')
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(self.out / 'training_curves.png')
        plt.close(fig)

    def summarize(self, manifest):
        import numpy as np
        import torch
        from experiments.training.fpga_road_seg.dataset_adapter import RoadDataset
        from experiments.training.fpga_road_seg.models import build_model
        from experiments.training.fpga_road_seg.profile import profile_model
        from experiments.training.fpga_road_seg.train import load_checkpoint

        rows = []
        report_rows = []
        for name in MODELS:
            ckpt = self.out / 'checkpoints' / name / 'best.pt'
            report_path = self.out / 'checkpoints' / name / 'report.json'
            eval_path = self.out / 'metrics' / f'{name}-heldout.json'
            if not (ckpt.exists() and report_path.exists() and eval_path.exists()):
                continue
            training = json.loads(report_path.read_text(encoding='utf-8'))
            evaluation = json.loads(eval_path.read_text(encoding='utf-8'))
            model, _ = load_checkpoint(ckpt, 'cuda')
            model.eval()
            shape = (1, 3, training['config']['height'], training['config']['width'])
            profile = profile_model(build_model(name), shape)
            dataset = RoadDataset(manifest, 'test', (training['config']['width'], training['config']['height']),
                                  baseline=name == 'twin-small')
            x = dataset[0][0][None].cuda()
            with torch.no_grad():
                for _ in range(12):
                    model(x)
                torch.cuda.synchronize()
                timing = []
                for _ in range(80):
                    start = time.perf_counter()
                    model(x)
                    torch.cuda.synchronize()
                    timing.append((time.perf_counter() - start) * 1000)
            cls = evaluation['classes']
            rec = dict(model=name, evidence='real BDD100K train + official val held-out',
                       train_samples=training['train_samples'], internal_val_samples=training['val_samples'],
                       heldout_samples=evaluation['samples'], parameters=profile['parameters'], macs=profile['macs'],
                       fp32_gpu_forward_ms_p50=float(np.percentile(timing, 50)),
                       fp32_gpu_forward_ms_p95=float(np.percentile(timing, 95)),
                       drivable_iou=cls.get('drivable_area', {}).get('iou'),
                       solid_f1=cls.get('solid_lane', {}).get('f1'),
                       dashed_f1=cls.get('dashed_lane', {}).get('f1'),
                       lane_iou=cls.get('lane', {}).get('iou'),
                       crosswalk_dense_iou=None, crosswalk_dense_recall=None,
                       crosswalk_metric_note='Unavailable: only BDD sparse positive curves; weak fine-tune is not dense-ground-truth evaluation')
            internal = min(training['history'], key=lambda e: e['validation']['loss'])['validation']['classes']
            # Full-stage best checkpoint uses the full validation split, whereas
            # pilot validation is smaller and has a separately selected best.
            full_history = [e for e in training['history'] if e.get('train_samples') == training['train_samples']]
            if full_history:
                internal = min(full_history, key=lambda e: e['validation']['loss'])['validation']['classes']
            rec['internal_val_drivable_iou'] = internal.get('drivable_area', {}).get('iou')
            rec['full_training_epochs'] = len(full_history)
            rec['training_forward_backward_seconds'] = sum(e['train_seconds'] for e in training['history'])
            rows.append(rec)
            report_rows.append(dict(summary=rec, evaluation=evaluation, training=training, profile=profile))
            torch.cuda.empty_cache()
        if rows:
            fields = list(rows[0])
            with open(self.out / 'metrics.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            (self.out / 'model_reports.json').write_text(json.dumps(report_rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            best = max((r for r in rows if r['internal_val_drivable_iou'] is not None), key=lambda r: r['internal_val_drivable_iou'])
            best_name = best['model']
            export_dir = self.out / 'exports' / best_name
            if not (export_dir / 'export-report.json').exists():
                self.run([sys.executable, '-m', 'experiments.training.fpga_road_seg.export',
                          '--checkpoint', self.out / 'checkpoints' / best_name / 'best.pt',
                          '--manifest', manifest, '--output', export_dir, '--calibration-count', '256',
                          '--compare-split', 'test'])
            return best_name
        return None

    def write_readme(self, manifest):
        index = self.out / 'data' / 'bdd_index.summary.json'
        summary = json.loads(index.read_text(encoding='utf-8')) if index.exists() else {}
        metrics = self.out / 'metrics.csv'
        metric_table = metrics.read_text(encoding='utf-8') if metrics.exists() else '(training/evaluation has not completed)'
        body = f'''# BDD100K 道路分割训练结果\n\n状态：真实 BDD100K 原图和官方 JSON 标注；train 按原始视频 ID 分组留出内部验证集，官方 val 作为独立 held-out test。结果没有复用 PR #4/B-2a 的模型权重或精度。\n\n## 数据\n\n- 数据来源：Internet Archive 公共 BDD100K 原包副本，原始 BDD100K 非商业教育/研究许可。官方下载在当前网络环境不可用时使用此公开副本；下载器对完整 ZIP 校验 MD5 并支持 Range 分块续传。\n- 划分摘要：`{json.dumps(summary, ensure_ascii=False)}`\n- 每个原始 train/val JSON 与同 stem 图像配对；test 原图没有公开像素标签，不参与指标。\n- 车道实线/虚线由 lane poly2d 曲线和 style 属性渲染；10 px 原图线宽对应 PR #4 的 512x288 画布 4 px。\n- BDD crosswalk 只有稀疏曲线，本次仅生成正像素弱监督 mask；主模型核心训练排除该通道，单独的 weak-crosswalk 微调不报告 dense IoU/Recall。\n- 当前没有获准可用的 dense crosswalk 标签，故跨线道密集准确率为 N/A。\n\n## 模型实测\n\n`metrics.csv` 中每行均为本分支对应模型在本机 CUDA 重新训练后的 held-out 指标。MACs 只计 Conv/Linear/MatMul；GPU 延迟是固定已预处理输入的模型前向，不含图像解码/传输。\n\n```csv\n{metric_table.rstrip()}\n```\n\n## 结果目录\n\n- `checkpoints/`：最佳、last 与每 5 epoch checkpoint（Git 忽略）。\n- `exports/`：最佳模型 FP32 ONNX、INT8 QDQ 和量化对照。\n- `comparisons/`：真实 held-out 道路可视化；`failures/`：错误样例及失败原因。\n- `training_curves.png`：三模型 train/internal-val loss 曲线。\n- `run.log`：下载、mask 渲染、训练、评估与导出日志。\n\n硬件兼容结论：候选网络使用 Conv/Depthwise/Pointwise、ReLU、Add 与 nearest Resize；TwinLiteNet+ Small 还包含 PReLU/注意力/转置卷积等算子。PG2L200H 实际支持的算子、定点格式、资源、时序、吞吐与 mask 回传接口均等待队友硬件资料确认；ONNX/QDQ 不是 FPGA 权重文件，本地没有板端实测。\n'''
        (self.out / 'README_RESULT.md').write_text(body, encoding='utf-8')
        self.run([sys.executable, '-m', 'experiments.training.fpga_road_seg.deliver_results',
                  '--output', self.out, '--manifest', manifest])

    def run_all(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'checkpoints').mkdir(exist_ok=True)
        (self.out / 'exports').mkdir(exist_ok=True)
        (self.out / 'comparisons').mkdir(exist_ok=True)
        (self.out / 'failures').mkdir(exist_ok=True)
        images, labels = self.wait_archives()
        index, manifest = self.prepare(images, labels)
        self.run([sys.executable, '-m', 'experiments.training.fpga_road_seg.visualize_masks',
                  '--manifest', manifest, '--output', self.out / 'comparisons' / 'bdd_gt_masks.png', '--count', '12'])
        tempbase = self.out / 'tmp' / 'pytest'
        tempbase.parent.mkdir(parents=True, exist_ok=True)
        self.run([sys.executable, '-m', 'pytest', 'experiments/training/fpga_road_seg/tests', '-q', '--basetemp', tempbase])
        completed = []
        for name, config in MODELS.items():
            try:
                self.train_model(name, config, manifest, pilot=True)
            except Exception as exc:
                self.record_failure(name + '-pilot', exc)
        for name, config in MODELS.items():
            try:
                checkpoint_dir = self.train_model(name, config, manifest)
                completed.append(name)
            except Exception as exc:
                self.record_failure(name, exc)
                continue
            self.plot_curves()
        self.note(f'model_training_complete={completed}')
        for name in completed:
            try:
                self.weak_crosswalk_experiment(name, MODELS[name], manifest, self.out / 'checkpoints' / name)
            except Exception as exc:
                self.record_failure(name + '-weak-crosswalk', exc)
        best = None
        try:
            best = self.summarize(manifest)
        except Exception as exc:
            self.record_failure('summary-export', exc)
        self.write_readme(manifest)
        self.note(f'pipeline_complete best={best}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--postprocess-only', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    data = Path(os.environ.get('BDD100K_DATA', r'C:\Users\JingyuWang\datasets\bdd100k_issue15'))
    out = Path(os.environ.get('BDD100K_OUTPUT', repo / 'outputs' / 'fpga_road_seg_final'))
    epochs = int(os.environ.get('BDD_SEG_EPOCHS', '12'))
    if epochs < 2:
        raise ValueError('BDD_SEG_EPOCHS must be >= 2')
    pipeline = Pipeline(repo, data, out, epochs)
    if args.postprocess_only:
        manifest = data / 'rendered_masks_w10_v2' / 'manifest.jsonl'
        pipeline.plot_curves()
        pipeline.summarize(manifest)
        pipeline.write_readme(manifest)
        pipeline.note('postprocess_complete')
    else:
        pipeline.run_all()


if __name__ == '__main__':
    main()
