"""Build a Chinese result report and aligned real-road model comparisons."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import CLASSES
from .dataset_adapter import RoadDataset, letterbox, restore, read_json, file_hash
from .infer import overlay
from .train import load_checkpoint, names_for


MODELS = ('candidate', 'candidate-plus', 'twin-small')


def number(value, precision=4):
    return 'N/A' if value is None or value == '' else f'{float(value):.{precision}f}'


def comparisons(out, manifest):
    checkpoints = {n: out/'checkpoints'/n/'best.pt' for n in MODELS}
    available = {n: load_checkpoint(p, 'cuda') for n,p in checkpoints.items() if p.exists()}
    if not available:
        return
    for model, _ in available.values():
        model.eval()
    dataset = RoadDataset(manifest, 'test')
    for i in np.linspace(0, len(dataset)-1, min(8,len(dataset)), dtype=int):
        row = dataset.rows[int(i)]
        image = np.asarray(Image.open(row['image']).convert('RGB'))
        with np.load(row['masks'], allow_pickle=False) as m:
            target = m['target'].astype(np.float32)
        target[3] = 0
        panels = [('Input', Image.fromarray(image)),
                  ('BDD rendered GT (10px native lines)', overlay(image,target,CLASSES,'GT / crosswalk excluded'))]
        for name, (model, checkpoint) in available.items():
            cfg = checkpoint['config']
            x, geometry = letterbox(image, size=(cfg['width'],cfg['height']))
            with torch.no_grad():
                prob = model(x[None].cuda()).sigmoid()[0].cpu().numpy()
            output_names = names_for(cfg['model'])
            for channel in checkpoint.get('excluded_channels', ()):
                if channel in output_names:
                    prob[output_names.index(channel)] = 0
            panels.append((name, overlay(image,restore(prob,geometry),names_for(name),name)))
        w,h,caption = 640,360,26
        canvas = Image.new('RGB',(w*3,(h+caption)*2),'white')
        draw = ImageDraw.Draw(canvas)
        for j,(title,panel) in enumerate(panels):
            x,y=(j%3)*w,(j//3)*(h+caption)
            draw.text((x+4,y+4),title,fill='black')
            canvas.paste(panel.resize((w,h),Image.Resampling.BILINEAR),(x,y+caption))
        draw.text((w*2+6,h+caption+10),f"held-out: {row['id']}",fill='black')
        folder=out/'comparisons'/'all_models'
        folder.mkdir(parents=True,exist_ok=True)
        canvas.save(folder/f"{row['id']}.jpg",quality=90)
    del available
    torch.cuda.empty_cache()


def report(out, manifest):
    metrics=out/'metrics.csv'
    rows=list(csv.DictReader(metrics.open(encoding='utf-8'))) if metrics.exists() else []
    summaries={r['model']:r for r in rows}
    training={n:read_json(out/'checkpoints'/n/'report.json') for n in MODELS
              if (out/'checkpoints'/n/'report.json').exists()}
    candidate_scores={}
    for name in ('candidate','candidate-plus'):
        if name not in training or name not in summaries:
            continue
        run=training[name]
        full=[e for e in run['history'] if e.get('train_samples')==run['train_samples']]
        if not full:
            continue
        classes=min(full,key=lambda e:e['validation']['loss'])['validation']['classes']
        vals=[classes[c][k] for c,k in (('drivable_area','iou'),('solid_lane','f1'),('dashed_lane','f1'))]
        if all(v is not None for v in vals):
            candidate_scores[name]=sum(vals)/3
    recommended=max(candidate_scores,key=candidate_scores.get) if candidate_scores else None
    table=['| 模型 | 参数 | MACs | drivable IoU | solid F1 | dashed F1 | lane IoU | GPU p50/p95 ms |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        r=summaries.get(name,{})
        table.append(f"| {name} | {r.get('parameters','N/A')} | {r.get('macs','N/A')} | "
                     f"{number(r.get('drivable_iou'))} | {number(r.get('solid_f1'))} | "
                     f"{number(r.get('dashed_f1'))} | {number(r.get('lane_iou'))} | "
                     f"{number(r.get('fp32_gpu_forward_ms_p50'),2)}/{number(r.get('fp32_gpu_forward_ms_p95'),2)} |")
    quant=[]
    for path in sorted((out/'exports').glob('*/export-report.json')):
        r=read_json(path)
        q=r['comparison']
        quant.extend([f"### {r['model']} INT8 PTQ/QDQ",
            f"校准：{r['calibration_samples']} 张 train 图像；比较：{q['samples']} 张 {q['split']}。",
            '| 类别 | FP32 IoU | INT8 IoU | IoU 差值 | FP32 F1 | INT8 F1 |',
            '|---|---:|---:|---:|---:|---:|'])
        for name,a in q['fp32'].items():
            if name=='crosswalk':
                continue
            b=q['int8'][name]
            delta=b['iou']-a['iou'] if a['iou'] is not None and b['iou'] is not None else None
            quant.append(f"| {name} | {number(a['iou'])} | {number(b['iou'])} | {number(delta)} | {number(a['f1'])} | {number(b['f1'])} |")
        quant.extend([f"PyTorch/FP32 ONNX 最大 logit 差 {q['torch_onnx_max_logit_error']:.6g}；"
                      f"FP32/INT8 有效像素阈值一致率 {q['threshold_mask_agreement_valid_pixels']:.4%}。",
                      f"FP32 实际算子：`{json.dumps(r['fp32_operators'])}`。",
                      '仅 Conv 权重/激活为 W8A8；Add、Resize、I/O 等仍浮点。该 ONNX 不能直接烧写 FPGA。',''])
    audit_path=Path(manifest).with_name('task-availability-audit.json')
    audit=read_json(audit_path) if audit_path.exists() else {}
    evidence=[]
    for name,run in training.items():
        full=[e for e in run['history'] if e.get('train_samples')==run['train_samples']]
        evidence.append(f"- {name}：{len(full)} 个当前规模 epoch，train={run['train_samples']}、internal-val={run['val_samples']}，"
                        f"AMP={run['amp']}，GPU={run['gpu']}；checkpoint SHA-256 `{file_hash(out/'checkpoints'/name/'best.pt')}`。")
    status='三模型真实训练与 held-out 评估完成' if len(rows)==3 else '部分结果；尚未完成的项保持 N/A'
    recommendation=(f'软件交接优先候选：**{recommended}**。在两个能直接输出实/虚线的 FPGA-friendly 候选中，'
                    '按内部验证的 drivable IoU、solid F1、dashed F1 三项平均值选择，未使用 held-out 指标选模型。'
                    '最终硬件推荐需通过下列板端适配条件。') if recommended else '尚无足够的完整验证结果来推荐模型。'
    body=f'''# FPGA 道路分割真实结果

状态：{status}。所有精度来自本分支新模型、新 checkpoint，没有复制 PR #4/B-2a 精度或权重。

## 数据与监督

完整恢复 BDD100K 100K Images 与每图 JSON，70k train、10k val 一一配对；20k test 无公开真值。
数据来源、大小和 MD5 见 `reports/real_bdd_recovery.md` 与本地下载 manifest；公开镜像不是官方签名背书。
从 JSON 的 poly2d 渲染原生 Mask，10px 原图线宽；direct/alternative 合并，solid/dashed 根据属性区分。
62,984 张训练，7,012 张内部验证，10,000 张官方 val 用作 held-out。与官方 val 视频前缀重叠的 4 张 train 帧隔离，
其原图、JSON 和 Mask 均保留；3 对同 split 相同 JPEG 保留，没有跨 split 相同内容。
混合检测 JSON 中缺失某项道路标注时，对应有效性为零，不生成全背景监督。核查：`{json.dumps(audit,ensure_ascii=False)}`。

crosswalk 仅有 BDD 稀疏曲线：主模型屏蔽该通道；独立 weak-crosswalk checkpoint 为正像素弱监督试验，
正监督存在全阳性退化解，不能据此证明定位能力。Mapillary Vistas 的密集斑马线标签尚未获得。
**全部 dense crosswalk IoU/Recall 为 N/A**，也没有伪造全背景斑马线标签。

## 真实 held-out 指标

{chr(10).join(table)}

IoU 是前景 IoU；F1 在下采样后的有效像素上按全数据 TP/FP/FN 累计。
320×192 RGB letterbox，固定阈值 0.5，padding 无效。线宽影响细线指标，不能直接与旧 B-2a 不同口径对比。
TwinLiteNet+ Small 保留原始两头结构，只输出 drivable/lane，solid/dashed F1 没有定义。
MACs 是 Conv/ConvTranspose/Linear/MatMul 的软件计算量，不包含所有逐元素开销。
GPU 延迟是已预处理 batch=1 FP32 模型前向，不含解码、传输和 FPGA 时延。

## 训练证据

每模型先做 512/128 张真实 train/val 的 3 epoch 试训，再继续全量训练。
小样本与全量验证范围不同，最佳损失分别选择；history 逐 epoch 保留样本量。

{chr(10).join(evidence)}

最佳权重仅由内部验证 loss 选择。`training_curves.png` 包含试训到全量的边界，边界处损失不可直接视为回退。
历史无效监督试训已移入 `failures/invalid_pilots_before_task_availability_fix/`，不参与当前训练、指标和导出。

## 推荐与硬件依赖

{recommendation}

Candidate 系列：Conv（普通/Depthwise/Pointwise）、ReLU、Add、nearest Resize 等。
TwinLiteNet+ Small 还含 PReLU、dilated Conv、ConvTranspose、CAAM 注意力、MatMul/Linear/Softmax、池化、reshape/transpose。
软件导出成功不等于 PG2L200H 支持；LUT/DSP/BRAM、目标时钟/时序、最大张量、dilation/resize 语义、
INT8 累加/四舍五入/饱和、权重打包格式和传输协议仍需队友确认。板端 FPS/功耗/时延没有实测。
交接接口与逐层校验步骤见 `docs/fpga_road_seg_handoff.md`。现有 FAST-Q4、PCIe、HUD、Tesla BEV 和 FPGA 工程未修改。

## 导出与量化对照

{chr(10).join(quant) if quant else '导出或量化比较尚未完成；见 failures/ 与 run.log，不能填写预计精度。'}

## 文件

- `metrics.csv`：三模型实际指标；空值表示无定义/无监督。
- `training_curves.png`：真实训练/内部验证曲线。
- `comparisons/all_models/`：同一 held-out 图像的输入、GT 和三模型预测。
- `failures/<model>/`：固定前 512 张 held-out 中的低 IoU 案例及 TP/FP/FN；这是子集筛选，未覆盖全体错误。
- `checkpoints/<model>/`：PyTorch best/last/周期 checkpoint；`pilot/` 为有效试训。
- `exports/`：FP32 ONNX、INT8 QDQ、校准清单、算子和量化参数、精度对照。
- `run.log`：运行日志；旧失败保留以供审计，不能将重试前的损失混入本次结果。

数据、权重及完整日志留在独立 outputs/data 目录，不提交到普通 Git。
'''
    (out/'README_RESULT.md').write_text(body,encoding='utf-8')
    return recommended


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--manifest',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    torch.set_num_threads(4)
    recommended=report(out,args.manifest)
    if recommended:
        from .export import export
        export(out/'checkpoints'/recommended/'best.pt',args.manifest,out/'exports'/recommended,
               calibration_count=256,compare_split='test')
        report(out,args.manifest)
    comparisons(out,args.manifest)
    print(json.dumps({'recommended_candidate':recommended,'output':str(out)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
