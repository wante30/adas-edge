"""Deliver measured stage-two comparisons, PTQ search, real images/video and handoff."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from . import CLASSES
from .dataset_adapter import RoadDataset, file_hash, letterbox, read_json, restore, write_json
from .engine import Metrics, evaluate
from .infer import overlay
from .profile import profile_model
from .quantize_stage2 import quantize
from .train import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]


def domain_datasets(manifest):
    domains = {}
    for source, split in [('bdd100k','test'),('rlmd','val')]:
        ds = RoadDataset(manifest,split)
        ds.rows = [r for r in ds.rows if r['source']==source]
        domains[source] = ds
    return domains


def measured_eval(model, domains):
    return {source: evaluate(model, DataLoader(ds,batch_size=32,num_workers=2,pin_memory=True), 'cuda', CLASSES)
            for source,ds in domains.items()}


@torch.no_grad()
def prediction(model, image):
    x,g = letterbox(image)
    return restore(model(x[None].cuda()).sigmoid()[0].cpu().numpy(),g)


def panel(row, before, after, dest):
    image = np.asarray(Image.open(row['image']).convert('RGB'))
    with np.load(row['masks']) as m:
        gt = m['target'].copy()
    for channel in row.get('disabled_channels',[]):
        gt[CLASSES.index(channel)] = 0
    a, b = prediction(before,image), prediction(after,image)
    a[3] = 0  # Stage-one crosswalk was excluded; do not show it as a trained head.
    panels = [('Input / '+row['source'], Image.fromarray(image)),
              ('GT / missing tasks hidden', overlay(image,gt,CLASSES,'Ground truth')),
              ('Stage1 / crosswalk untrained', overlay(image,a,CLASSES,'Before')),
              ('Stage2 / all four predictions', overlay(image,b,CLASSES,'After'))]
    canvas = Image.new('RGB',(1280,2*386),'white'); draw=ImageDraw.Draw(canvas)
    for i,(caption,im) in enumerate(panels):
        x,y=i%2*640,i//2*386
        draw.text((x+5,y+5),caption+' '+row['id'],fill='black')
        canvas.paste(im.resize((640,360)),(x,y+26))
    canvas.save(dest,quality=90)


@torch.no_grad()
def visuals(out, domains, before, after):
    folder=out/'comparisons'; folder.mkdir(exist_ok=True)
    failures=out/'failures'; failures.mkdir(exist_ok=True)
    bad=[]
    for source,ds in domains.items():
        rows=sorted(ds.rows,key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        if source=='rlmd':
            rows=([r for r in rows if r['positive_pixels'][3]>0][:8] +
                  [r for r in rows if r['positive_pixels'][3]==0][:4])
        else:
            rows=rows[:8]
        for row in rows:
            panel(row,before,after,folder/f"{row['id']}.jpg")
        # Evaluate every RLMD val sample for actual FN/FP ranking. BDD's complete
        # domain metrics are separate; this error gallery is targeted at crosswalk.
        if source=='rlmd':
            for i,row in enumerate(ds.rows):
                x,y,v=ds[i]; metric=Metrics(4)
                prob=after(x[None].cuda()).sigmoid().cpu()
                metric.update(prob,y[None],v[None])
                cw=metric.report(CLASSES)['crosswalk']
                if cw['tp']+cw['fn'] > 0 or cw['fp']>40:
                    bad.append(dict(id=row['id'],crosswalk=cw))
            by_id={r['id']:r for r in ds.rows}
            bad.sort(key=lambda r: (r['crosswalk']['iou'] or 0, -r['crosswalk']['fn']-r['crosswalk']['fp']))
            for item in bad[:12]:
                panel(by_id[item['id']],before,after,failures/f"{item['id']}.jpg")
    write_json(failures/'crosswalk_errors.json',bad)
    (failures/'README.md').write_text('错误按真实验证集逐像素 IoU 排序。FN 为漏检、FP 为误检。不能仅据这些图判断昼夜/遮挡的因果，视频无真值不报告精度。\n',encoding='utf-8')


@torch.no_grad()
def real_video(out,before,after):
    source=Path('D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/video_20260929_113421.mp4')
    if not source.is_file():
        write_json(out/'failures/video.json',dict(reason='Original local reference video unavailable; annotated historical output is not used as raw input'))
        return None
    cap=cv2.VideoCapture(str(source)); fps=cap.get(cv2.CAP_PROP_FPS) or 30
    writer=cv2.VideoWriter(str(out/'comparisons/real_video_before_after.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,360))
    if not writer.isOpened():
        raise RuntimeError('MP4 writer unavailable')
    frames=0
    while frames < min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),round(fps*15)):
        ok,bgr=cap.read()
        if not ok: break
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        a,b=prediction(before,rgb),prediction(after,rgb);a[3]=0
        panels=[Image.fromarray(rgb),overlay(rgb,a,CLASSES,'Stage1 (CW untrained)'),overlay(rgb,b,CLASSES,'Stage2')]
        frame=np.concatenate([np.asarray(p.resize((640,360))) for p in panels],axis=1)
        writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        if frames in (0,90,180,270,360):
            Image.fromarray(frame).save(out/'comparisons'/f'video_frame_{frames:04d}.jpg')
        frames+=1
    cap.release();writer.release()
    record=dict(source=str(source),sha256=file_hash(source),frames=frames,fps=fps,ground_truth=False,
                note='Real contiguous original video, qualitative inspection only. No video accuracy or FPGA timing claim.')
    write_json(out/'comparisons/video.json',record)
    return record


def curves(out, history):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    epoch=[e['epoch'] for e in history]
    axes[0,0].plot(epoch,[e['train_loss'] for e in history],label='mixed training')
    axes[0,0].plot(epoch,[e['validation']['loss'] for e in history],label='domain mean validation')
    for c,m in [('drivable_area','iou'),('solid_lane','f1'),('dashed_lane','f1')]:
        axes[0,1].plot(epoch,[e['validation']['domains']['bdd100k']['classes'][c][m] for e in history],label=c+' '+m)
    for m in ('iou','f1','recall'):
        axes[1,0].plot(epoch,[e['validation']['domains']['rlmd']['classes']['crosswalk'][m] for e in history],label='crosswalk '+m)
    axes[1,1].plot(epoch,[e['selection_score'] for e in history],label='constrained selection score')
    for ax in axes.flat:
        ax.set_xlabel('Mixed sampling epoch');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)


@torch.no_grad()
def latency(model):
    x=torch.zeros(1,3,192,320,device='cuda')
    for _ in range(30):model(x)
    torch.cuda.synchronize();samples=[]
    for _ in range(200):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();model(x);b.record();torch.cuda.synchronize();samples.append(a.elapsed_time(b))
    return dict(device=torch.cuda.get_device_name(),batch=1,input=[1,3,192,320],precision='FP32',
                warmup=30,runs=200,p50_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),
                scope='GPU model forward only, excludes preprocessing/H2D/display; not FPGA latency')


def metric_rows(reports, profile, timing):
    rows=[]
    for name,domains in reports.items():
        for source,domain in domains.items():
            for cls,metrics in domain['classes'].items():
                rows.append(dict(model='candidate-plus',variant=name,dataset=source,
                    split='official_val_heldout' if source=='bdd100k' else 'independent_group_val',
                    samples=domain['samples'],class_name=cls,**metrics,parameters=profile['parameters'],macs=profile['macs'],
                    fp32_gpu_p50_ms=timing['p50_ms'],fp32_gpu_p95_ms=timing['p95_ms']))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--force-qat',action='store_true',help='bounded QAT in a new output; preserve previous PTQ result')
    p.add_argument('--report-only',action='store_true',help='reuse verified evaluation/visuals/export; refresh reports without retraining')
    a=p.parse_args();out=Path(a.output);torch.set_num_threads(4)
    from .refine_stage2 import maybe_refine
    refinement=maybe_refine(out,a.manifest)
    best=out/'checkpoints/candidate-plus/best.pt'
    if not best.exists():
        (out/'README_RESULT.md').write_text('# 第二阶段未通过内部验证选择\n\n没有 checkpoint 同时改善混合分数且通过 BDD 三项退化保护。保留第一阶段模型。真实历史见 checkpoints/candidate-plus/report.json；不得将最后权重冒充有效最终模型。下一步需分析类别冲突并尝试轻量蒸馏。\n',encoding='utf-8')
        raise RuntimeError('No eligible improved checkpoint; FP32/INT8 acceptance not reached')
    before,old=load_checkpoint(a.baseline,'cuda');after,ck=load_checkpoint(best,'cuda')
    before.eval();after.eval();domains=domain_datasets(a.manifest)
    evaluation=out/'evaluation.json'
    if evaluation.exists():
        measured=read_json(evaluation)
        if measured['checkpoint_sha256']!=file_hash(best):raise ValueError('stale evaluation')
    else:
        measured=dict(checkpoint_sha256=file_hash(best),baseline_sha256=file_hash(a.baseline),manifest_sha256=file_hash(a.manifest),
                      before=measured_eval(before,domains),after=measured_eval(after,domains))
        write_json(evaluation,measured)
    print('completed before/after real domain evaluation',flush=True)
    timing=read_json(out/'latency.json') if a.report_only else latency(after)
    write_json(out/'latency.json',timing)
    profile=profile_model(load_checkpoint(best)[0],(1,3,192,320));write_json(out/'profile.json',profile)
    history=read_json(out/'checkpoints/candidate-plus/report.json')['history']
    mixed_epochs=len(history)
    if refinement.get('triggered') and (out/'checkpoints/candidate-plus-distillation/report.json').exists():
        extra=read_json(out/'checkpoints/candidate-plus-distillation/report.json')['history']
        history=history+[dict(e,epoch=mixed_epochs+e['epoch']) for e in extra]
    if a.report_only:
        curves(out,history)
        video=read_json(out/'comparisons/video.json') if (out/'comparisons/video.json').exists() else None
    else:
        curves(out,history);visuals(out,domains,before,after);video=real_video(out,before,after)
    print('completed real-road image and video artifacts',flush=True)
    quant=quantize(load_checkpoint(best)[0],best,a.manifest,out/'exports',ck['config'],force_qat=a.force_qat)
    from .inspect_quantization import inspect as inspect_quantized_predictions
    quant_timing=inspect_quantized_predictions(out,a.manifest)
    if quant['failures']:
        write_json(out/'failures/quantization.json',quant['failures'])
    reports=dict(before_fp32=measured['before'],after_fp32=measured['after'],after_onnx_fp32=quant['comparison']['fp32'],after_int8=quant['comparison']['int8'])
    rows=metric_rows(reports,profile,timing)
    with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'metrics.json',rows)
    table=['| 数据集 / 指标 | 微调前 FP32 | 微调后 FP32 | INT8 |','|---|---:|---:|---:|']
    deltas=[]
    keys=[('bdd100k','drivable_area','iou'),('bdd100k','solid_lane','f1'),('bdd100k','dashed_lane','f1'),
          ('rlmd','crosswalk','iou'),('rlmd','crosswalk','f1'),('rlmd','crosswalk','recall')]
    for source,cls,m in keys:
        oldv=measured['before'][source]['classes'][cls][m];newv=measured['after'][source]['classes'][cls][m]
        q=quant['comparison']['int8'][source]['classes'][cls][m]
        def pct(v):return 'N/A' if v is None else f'{v*100:.2f}%'
        table.append(f'| {source} {cls} {m} | {pct(oldv)} | {pct(newv)} | {pct(q)} |')
        if newv is not None and q is not None:deltas.append(newv-q)
    drop=max(deltas)
    accepted=drop<=.02
    checkpoint_sha=file_hash(best)
    audit=read_json(Path(a.manifest).parent/'audit.json')
    content=f'''# 第二阶段真实训练结果

Candidate-plus 从原最佳权重继续混合微调，网络未改变：{profile['parameters']:,} 参数、{profile['macs']:,} MACs（320×192，Conv 算术统计）。
第一阶段所有模型和量化产物保留。本轮实际完成 {mixed_epochs} 个混合采样 epoch，附加蒸馏 {len(history)-mixed_epochs} 轮。
当前选择{'蒸馏分支' if refinement.get('selected') else '混合微调分支'}最佳 epoch {ck['epoch']}，分支比较见 refinement.json。

{chr(10).join(table)}

BDD 表格是 10,000 张官方 val held-out；模型选择使用另有 7,012 张内部验证。
RLMD 是 {audit['split_counts']['val']} 张独立分组验证，训练 {audit['split_counts']['train']} 张；没有额外独立 RLMD 测试集，不能把它称为测试精度。
前模型 crosswalk 未训练，数值仅用于展示未训练输出，不是旧模型有效斑马线精度。
RLMD 标注覆盖整个过街区域、含条纹间隙；未标 drivable 的 loss 始终屏蔽。
BDD 旧曲线 crosswalk 在本轮屏蔽。数字文件名分组和近似图像排查减少邻帧泄漏，但原数据缺少完整录像 ID，仍不能证明场景完全独立。
所有分割指标在 320×192 letterbox 网格上统计，padding 不计分；不是原始分辨率逐像素评分。

## 量化对照

先完成 FP32 微调，再在真实 TRAIN 样本上进行 {sum(v['name']!='qat' for v in quant['variants'])} 种 PTQ 对照。
校准 {quant['calibration_samples']} 张，BDD / RLMD 平衡且包含 crosswalk 正负样本；不用验证/测试图校准。
量化器选择使用内部验证，选择 `{quant['selected']}`。完整候选结果见 exports/search.json。
PyTorch / ONNX 最大 logit 误差 {quant['torch_onnx_max_logit_error']:.6g}。
PTQ 未通过内部逐项 2 点保护时，自动尝试最多 3 轮真实混合数据 QAT（冻结 BN/校准范围，FP32 logits 蒸馏正则），结果及异常均记录于 exports/。
本轮显式要求 QAT：{a.force_qat}；实际 QAT 是否成功、是否被采用见 exports/search.json，不能把已请求等同于已成功。
若本轮从之前已评估 PTQ 的结果继续，先前的留出集结果已经看过，后续同集合复评不能称为新的盲测；量化参数/候选选择仍只使用训练校准和内部验证。
表中关键 IoU/F1/Recall 最大量化下降 {drop*100:.2f} 个百分点；2 点软件门槛：{'通过' if accepted else '未通过，仍需逐层误差定位或进一步 QAT，不能宣布 INT8 可用'}。
ONNX 只有 Conv W8A8，Add / Resize / I/O 仍浮点，不能直接烧录 PG2L200H。
如选用 unsigned asymmetric 激活，RTL 需明确支持零点修正。QDQ 文件不是队友硬件权重格式。

## 真实道路检查与性能

comparisons/ 包含真实道路 GT、微调前后对照、原始连续视频预测；failures/ 按真实 RLMD crosswalk IoU 排序并记录 FP/FN。
本地原始视频{'已生成连续预测，详情见 comparisons/video.json' if video else '不可用，原因见 failures/video.json'}，没有逐像素视频真值，不报告视频精度。
RTX 4060 FP32 batch1 前向 p50/p95：{timing['p50_ms']:.3f}/{timing['p95_ms']:.3f} ms；不含预处理/传输/绘图，不是 FPGA 延迟。
CPU ONNX Runtime（4 线程，真实道路输入、20 次预热/100 次测量）前向 p50：FP32 {quant_timing['fp32']['p50_ms']:.3f} ms，
INT8 {quant_timing['int8']['p50_ms']:.3f} ms。这是该 CPU/运行时的实测，不能推断 INT8 必然更快或推断 FPGA 吞吐；
comparisons/quantization/ 提供 FP32/INT8/GT 和阈值分歧图，quantization_comparison.png 为内部验证候选对照。

## 交接与限制

best checkpoint SHA-256：`{checkpoint_sha}`。训练/优化器/AMP/随机状态保存在 checkpoints/，可恢复。
官方 RLMD 无明确许可证；公开副本 MIT 声明未获原作者确认。本地科研实验，数据与预测对照均不随代码提交。
权重、原数据、视频不进普通 Git。详细算子及接口合同见 FPGA_HANDOFF.md。
本轮仅形成待硬件验收的软件候选；是否满足道路实际可用性需结合错误图及场景验证，不能由 loss 单独断言。
中间真实图发现 RLMD 仪表台区域 drivable 误检和斑马线碎片；RLMD 无 drivable 真值，不能为消除误检而伪造背景监督。未提供这类场景的可行驶区域实测精度。
最终失败图 rlmd-28180 的雨天过街区域完全漏检（对应样本 crosswalk IoU=0）；视频只做抽帧定性检查，不能把保存预测视频等同于验证连续稳定性。
给 PG2L200H 的建议：保留 Candidate-plus FP32 为当前四通道软件基线，先确认算子和量化合同，再做板端逐层对齐；现有斑马线质量不足以宣布模型已满足实际道路使用要求。
'''
    (out/'README_RESULT.md').write_text(content,encoding='utf-8')
    handoff=f'''# PG2L200H 第二阶段交接

- 架构未改：Conv 3×3 / depthwise / pointwise，BN 导出折叠，ReLU、残差 Add、nearest Resize；无注意力。
- 输入 RGB FP32 NCHW [1,3,192,320]；uint8/255；等比 letterbox，填 114；标签 padding 不计分。
- 输出 logits [1,4,192,320]，顺序 drivable_area、solid_lane、dashed_lane、crosswalk；独立 sigmoid≥0.5，可重叠，非互斥 softmax。
- 参数 {profile['parameters']}、MACs {profile['macs']}；模型校验和 {checkpoint_sha}。
- exports/fp32-operators.json / int8-operators.json 包含实际导出算子与 tensor 形状，quantization.json 包含逐层 scale、zero_point、int32 bias 与 multiplier/shift 近似。
- 量化器 {quant['selected']}，关键指标最大降 {drop*100:.2f} 点；{'软件量化门槛通过' if accepted else '软件量化门槛未通过，需继续 QAT/误差定位'}。
- 待确认：depthwise groups、各通道倍数、padding/stride、Add 跨支路 scale 对齐、nearest 坐标、INT32 累加溢出、舍入/饱和、非零 zero_point 修正、片上缓存/DDR/带宽、权重打包与 PCIe 协议。
- 当前 Conv QDQ 外仍有浮点，必须确认完整整数图适配；没有板端资源、频率、FPS、功耗、准确率实测。
- crosswalk 定义为过街区域（含条纹间隙），不能解释为单条白纹；数据许可状态见结果报告。
- 不涉及 FAST-Q4、PCIe、HUD、Tesla BEV、队友 FPGA 工程或队友模型权重修改。
'''
    (out/'FPGA_HANDOFF.md').write_text(handoff,encoding='utf-8')
    publish=ROOT/'experiments/training/fpga_road_seg/reports/stage2_results';publish.mkdir(parents=True,exist_ok=True)
    for name in ('README_RESULT.md','metrics.csv','metrics.json','training_curves.png','FPGA_HANDOFF.md','latency.json',
                 'quantization_comparison.png','quantization_inspection.json'):
        shutil.copyfile(out/name,publish/name)
    write_json(publish/'data-audit.json',audit)
    write_json(publish/'refinement.json',refinement)
    write_json(publish/'quantization_summary.json',dict(selected=quant['selected'],variants=quant['variants'],comparison=quant['comparison'],torch_onnx_max_logit_error=quant['torch_onnx_max_logit_error'],quantization_accepted=accepted))
    from .verify_delivery import verify
    verify(out)
    shutil.copyfile(out/'artifact_verification.json',publish/'artifact_verification.json')
    from .publish_stage2 import publish as publish_results
    publish_results(out)
    print('delivery_complete '+str(out/'README_RESULT.md'),flush=True)


if __name__=='__main__':main()
