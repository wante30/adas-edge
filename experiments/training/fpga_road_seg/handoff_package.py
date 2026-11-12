"""Finish compatibility entries, self-check and real ZIP/RAR extraction checks."""
import argparse
import contextlib
import datetime
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from .handoff_integer import load_json, sha256
from .handoff_verify import verify
from .handoff_documentation import documents


INFER = '''"""Run frozen production ONNX on FP32 NCHW golden input, no preprocessing."""
import argparse,json
from pathlib import Path
import numpy as np
import onnxruntime as ort
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True);p.add_argument('--input',required=True)
p.add_argument('--output',required=True);p.add_argument('--config',required=True)
a=p.parse_args();c=json.loads(Path(a.config).read_text(encoding='utf-8'))
x=np.fromfile(a.input,dtype='<f4').reshape(c['input_shape'])
o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1
s=ort.InferenceSession(a.model,o,providers=['CPUExecutionProvider'])
y=s.run(None,{'rgb':x})[0]
if list(y.shape)!=c['output_shape'] or not np.isfinite(y).all():raise ValueError('invalid model output')
with Path(a.output).open('xb') as f:f.write(y.astype('<f4').tobytes())
print(json.dumps(dict(shape=list(y.shape),dtype='float32',semantics='logits',backend='ORT CPU, default optimizations')))
'''

EXPORT = '''"""Copy the pinned, measured ONNX export; never silently retrace QAT."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
a=p.parse_args();root=Path(a.root);identity=json.loads((root/'MODEL_IDENTITY.json').read_text(encoding='utf-8'))
source=root/'model/model_int8.onnx'
if hashlib.sha256(source.read_bytes()).hexdigest()!=identity['sha256']['int8']:raise ValueError('wrong final ONNX')
with source.open('rb') as src,Path(a.output).open('xb') as dst:shutil.copyfileobj(src,dst)
print('Copied verified final QAT export. Full training/export provenance is in 06_参考推理与转换工具/source_provenance, requires the source repository; this script does not retrain or requantize.')
'''

QUANTIZE = '''"""Extract existing QAT Conv quantization from ONNX; no PTQ/recalibration."""
import argparse,json,math
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
m=onnx.load(a.model);init={i.name:numpy_helper.to_array(i) for i in m.graph.initializer}
prod={o:n for n in m.graph.node for o in n.output};users={}
for n in m.graph.node:
 for key in n.input:users.setdefault(key,[]).append(n)
rows=[]
for n in m.graph.node:
 if n.op_type!='Conv':continue
 xq,wq=prod[n.input[0]],prod[n.input[1]]
 w,sw,zw=[init[key] for key in wq.input]
 sx=float(init[xq.input[1]]);zx=int(init[xq.input[2]])
 consumers=users[n.output[0]];nextnode=consumers[0]
 if nextnode.op_type=='Relu':nextnode=users[nextnode.output[0]][0]
 assert nextnode.op_type=='QuantizeLinear'
 sy=float(init[nextnode.input[1]]);zy=int(init[nextnode.input[2]])
 b=init[n.input[2]];bscale=sx*sw.astype(np.float64)
 b32=np.rint(b.astype(np.float64)/bscale).astype('<i4')
 mantissa,exponent=np.frexp(bscale/sy);mult=np.rint(mantissa*(1<<30)).astype('<i4');shift=(30-exponent).astype('<i4')
 stem=f'conv_{len(rows):02d}'
 for suffix,arr in [('weight',w),('bias_int32',b32),('weight_scale',sw),('multiplier',mult),('right_shift',shift)]:arr.tofile(out/(stem+'.'+suffix+'.bin'))
 rows.append(dict(name=n.name,weight_shape=list(w.shape),weight_dtype=str(w.dtype),weight_scale=sw.tolist(),weight_zero_point=zw.tolist(),
  input_scale=sx,input_zero_point=zx,output_scale=sy,output_zero_point=zy,bias_int32=b32.tolist(),bias_scale=bscale.tolist(),
  multiplier=mult.tolist(),right_shift=shift.tolist()))
(out/'extracted.json').write_text(json.dumps(dict(method='extract existing QAT; proposed integer-v1 bias/requant; no new calibration',layers=rows),indent=2),encoding='utf-8')
print('Extracted',len(rows),'Conv layers; OIHW INT8, INT32 little-endian bias and fixed point parameters')
'''


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def compatibility(root):
    identity=load_json(root/'MODEL_IDENTITY.json');contract=load_json(root/'04_量化参数/integer_contract.json')
    summary=load_json(root/'03_网络结构与算子/architecture_summary.json')
    for name in ['model','config','golden','scripts']:(root/name).mkdir(exist_ok=True)
    for target,source in [('model_int8.onnx','model.int8.qdq.onnx'),('model.onnx','model.fp32.onnx'),('checkpoint.pth','qat.best.pt')]:
        shutil.copy2(root/'02_模型与权重'/source,root/'model'/target)
    shutil.copy2(root/'06_参考推理与转换工具/restore_checkpoint.py',root/'model/model.py')
    write_json(root/'config/model_config.json',dict(model_name='Candidate-plus',version=identity['version'],
        input_shape=[1,3,192,320],input_layout='NCHW',input_dtype='float32',output_shape=[1,4,192,320],
        output_dtype='float32',output_semantics='four independent logits, not argmax',num_classes=4,
        activation='uint8 in QAT Conv quantization; ONNX I/O float32',accumulator='int32 for derived integer-v1',
        integer_core_input_dtype='uint8',integer_core_output_dtype='uint8',threshold=.5,
        int8_onnx_sha256=identity['sha256']['int8'],checkpoint_restore='from model.model import restore_qat; model=restore_qat(path)'))
    write_json(root/'config/preprocess.json',dict(source_color='RGB',target_color='RGB',resize=[320,192],
        dimension_order='width,height',resize_mode='letterbox',image_interpolation='OpenCV INTER_LINEAR',crop=None,
        pad_value=[114,114,114],centering='left/top floor half; right/bottom receive remainder',scale=1/255,
        mean=[0,0,0],std=[1,1,1],input_dtype='float32',source_dtype='uint8',layout='NCHW',
        hardware_activation_quantization=contract['stages'][0],postprocess='sigmoid, unpad probabilities, bilinear restore, threshold>=0.5'))
    write_json(root/'config/quantization.json',dict(method='final selected QAT, frozen ranges',weight_dtype='int8',
        activation_dtype='uint8',weight_scheme='per_channel_symmetric axis0',activation_scheme='per_tensor_affine',
        source_graph_integer_scope='QDQ contains float Conv bias/Add/Resize/I/O; ORT CPU may fuse',
        integer_contract='04_量化参数/integer_contract.json',accumulator_dtype='int32',
        layers=contract['stages'],requantization='INT64 product, ties-even right shift, add zero_point, saturate uint8'))
    (root/'config/labels.txt').write_text('0 drivable_area\n1 solid_lane\n2 dashed_lane\n3 crosswalk\n',encoding='utf-8')
    rows=load_json(root/'05_硬件验证向量/samples.json')
    for number,row in enumerate(rows):
        folder=root/'05_硬件验证向量'/row['id']
        for target,source in [('input','input_fp32'),('output','output_qdq_logits_fp32')]:
            for extension in ['bin','npy']:
                shutil.copy2(folder/f'{source}.{extension}',root/'golden'/f'{target}_{number:03d}.{extension}')
        shutil.copy2(folder/'qdq_overlay.jpg',root/'golden'/f'output_{number:03d}.jpg')
    (root/'golden/README_GOLDEN.md').write_text('''# Golden 协议

input_NNN.bin：float32 little endian，连续 NCHW [1,3,192,320]，已 RGB letterbox 填114、/255，与 ONNX 输入相同。
output_NNN.bin：float32 little endian，连续 NCHW [1,4,192,320]，最终 QAT ONNX 默认 ORT CPU 的 logits，不是概率或类别 mask。
NPY 与 BIN 数据完全相同，JPG 是实际 QDQ 四类叠加。output 通道 0 drivable_area、1 solid_lane、2 dashed_lane、3 crosswalk。

这是生产 QDQ 软件 golden。RTL 全整数开发请使用 ../05_硬件验证向量/ 下 input_quantized_u8 和 integer/、accumulator/；不要将这里的 float32 字节当成 UINT8 核心输入。
真实来源/帧号/时间、所有中间层和量化参数见 ../05_硬件验证向量/samples.json、各 sample.json。两路径数值并非保证逐位相等。
''',encoding='utf-8')
    for name,text in [('infer.py',INFER),('export_onnx.py',EXPORT),('quantize.py',QUANTIZE)]:
        (root/'scripts'/name).write_text(text,encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('handoff_verify.py'),root/'06_参考推理与转换工具/handoff_verify.py')
    (root/'06_参考推理与转换工具/requirements.txt').write_text(
        f"numpy=={summary['runtime_versions']['numpy']}\nonnx=={summary['runtime_versions']['onnx']}\nonnxruntime=={summary['runtime_versions']['onnxruntime']}\ntorch=={summary['runtime_versions']['torch'].split('+')[0]}\n",encoding='utf-8')
    (root/'VERSION.txt').write_text(identity['version']+'\n',encoding='utf-8')


def finish(root,model_root,repo,rar_executable=None,make_archive=True):
    root,model_root,repo=[Path(v).resolve() for v in (root,model_root,repo)];compatibility(root)
    performance=root/'07_性能与可视化'
    for source in Path(__file__).parent.glob('handoff_*.py'):
        shutil.copy2(source,root/'06_参考推理与转换工具/source_provenance'/source.name)
    client_checks=repo/'outputs/handoff_client_checks_v2/client_checks.json'
    if client_checks.exists():
        shutil.copy2(client_checks,performance/'standalone_client_checks.json')
    for name in ['metrics.csv','evaluation.json','profile.json','latency.json','quantization_inspection.json','training_curves.png',
                 'qat_training_curves.png','quantization_comparison.png','README_RESULT.md']:
        shutil.copy2(model_root/name,performance/('stage2_'+name))
    shutil.copy2(model_root/'exports/search.json',performance/'quantization_search.json')
    video=repo/'outputs/fpga_road_seg_video_test'
    for name in ['README_RESULT.md','VISUAL_REVIEW.md','summary.json','quantization_video_difference.json']:
        shutil.copy2(video/name,performance/('video_'+name))
    failure=performance/'video_failures';failure.mkdir(exist_ok=True)
    for file in (video/'failures').iterdir():
        if file.is_file():shutil.copy2(file,failure/file.name)
    documents(root,model_root,repo)
    identity=load_json(root/'MODEL_IDENTITY.json')
    readme=f'''# Candidate-plus 最终 QAT 模型正式交接

版本：{identity['version']}。真实模型：32,524 参数、84,718,080 MACs，RGB 320×192，四个独立通道。

**队友第一步：解压后进入本目录，运行下列自检；然后从 s01 Conv 的 accumulator 开始做 RTL 对齐。**

```powershell
python -m pip install -r 06_参考推理与转换工具/requirements.txt
python 06_参考推理与转换工具/handoff_verify.py --root .
python 06_参考推理与转换工具/handoff_integer.py infer --root . --input 05_硬件验证向量/000_crosswalk_front/input_quantized_u8.npy --output ../integer_recheck_000
python scripts/infer.py --model model/model_int8.onnx --input golden/input_000.bin --output ../onnx_recheck_000.bin --config config/model_config.json
```

完整运行清单和架构、协议、全部量化规则见 [正式技术文档](01_技术文档/FPGA_CandidatePlus_技术交接文档.md)。

- 01_技术文档：正式规范、格式和接口待协商表。
- 02_模型与权重：最终 QAT、FP32、FX checkpoint、TorchScript、标准 OIHW 权重。
- 03_网络结构与算子：93 原始节点/20 激活 stage，完整 CSV/JSON、MAC/内存。
- 04_量化参数：20 激活域，732 输出通道 scale/zp/bias/requant，全部二进制。
- 05_硬件验证向量：6 个真实道路场景、输入输出、20 层 UINT8 和 14 层 INT32 acc。
- 06_参考推理与转换工具：可独立运行的 NumPy 整数推理、ORT 追踪、自检与来源。
- 07_性能与可视化：真实训练、验证、量化和视频失败证据。
- model/config/golden/scripts：所提供打包说明的便捷入口，数值按真实 Candidate-plus 填写。

QDQ 软件模型与全整数 integer-v1 是两个明确区分的目标，不能把软件 QDQ 宣称为已可烧录的 FPGA 模型。integer-v1 使用原 QAT INT8 权重/range，但新增 INT32 Bias、RNE/饱和和 Add 定点规则；六个真实样本已有差异实测，未跑全验证集迁移精度。

已知语义限制：白车、方向箭头和树木/建筑背景 crosswalk 误检，车道破碎/局部跳变，真实斑马线仍会漏检。FP32 同样可见，不能承诺道路稳定可用。

未确定硬件接口：总线/握手/寄存器/地址、lane 打包/对齐、DDR/BRAM/DSP 预算、实际时钟、tile/复用、预后处理责任、输出 logits 或 bitmask、性能及精度验收目标。标准 OIHW/NCHW 不代表队友现有 RTL 格式。

SHA256SUMS.txt 为 payload 校验清单，FILELIST.json 为逐文件目录；SELF_CHECK.json 是本次软件自检结果。归档和解压复核结果在外部 DELIVERY.json。

训练/导出环境和时间见 MODEL_IDENTITY.json、architecture_summary.json；原始导出时间不能仅靠文件 mtime 确认，本包生成时间不是模型训练时间。
最终 QAT ONNX SHA256：{identity['sha256']['int8']}。

`scripts/export_onnx.py` 复制并校验已冻结的最终导出，明确不会重新追踪改写已验证文件；`scripts/quantize.py` 从现有 QAT 图重新提取参数/标准权重，明确不会 PTQ 重校准。原训练/导出代码保留在 source_provenance，执行完整历史训练需要源仓库和原始数据，交接自检/整数推理不需要。

权重或工程资料缺失项：无。本包不包含 FPGA RTL/bitstream、PG2L200H 板测、完整训练数据和原视频，这些不属于已完成成果。模型名不是附件示例 tiny_unet；附加 RAR/ZIP 使用 CandidatePlus 命名。
'''
    (root/'README.md').write_text(readme,encoding='utf-8');(root/'README_MODEL.md').write_text(readme,encoding='utf-8')
    # Full self-check first, without checksums which are generated afterwards.
    with (performance/'checkpoint_restore.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result=verify(root,checksums=False)
    write_json(root/'SELF_CHECK.json',result)
    files=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name not in ['SHA256SUMS.txt','FILELIST.json'] and '__pycache__' not in path.parts:
            files.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=sha256(path)))
    (root/'SHA256SUMS.txt').write_text('\n'.join(f"{f['sha256']}  {f['path']}" for f in files)+'\n',encoding='utf-8')
    write_json(root/'FILELIST.json',dict(version=identity['version'],files=files,total_payload_bytes=sum(f['bytes'] for f in files),
                                     excluded_from_checksums=['SHA256SUMS.txt','FILELIST.json','runtime __pycache__']))
    print(f"SELF_CHECK {result['status']} arrays={result['parsed_arrays']} samples={len(result['samples'])}",flush=True)
    if not make_archive:return result
    stamp=identity['generated_utc'][:10].replace('-','')
    archive=root.with_name('CandidatePlus_PG2L200H_v1.0.0_'+stamp+'.zip')
    if archive.exists():raise FileExistsError('archive exists; preserve or select a new output name')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in [*files,dict(path='SHA256SUMS.txt'),dict(path='FILELIST.json')]:
            path=root/file['path'];z.write(path,root.name+'/'+file['path'])
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC check failed')
        with tempfile.TemporaryDirectory(prefix='fpga_zip_extract_',dir=root.parent) as temp:
            z.extractall(temp);extracted=verify(Path(temp)/root.name)
    delivery=dict(version=identity['version'],zip=dict(path=str(archive.resolve()),bytes=archive.stat().st_size,sha256=sha256(archive)),
                  software_self_check=result,zip_extracted_self_check=extracted)
    if rar_executable:
        rar=archive.with_suffix('.rar')
        if rar.exists():raise FileExistsError(rar)
        # The official RAR CLI is external tooling, not included in model payload.
        with (root.parent/'handoff_rar_pack.log').open('w',encoding='utf-8') as log:
            # Archive the ZIP payload, not the live directory: checkpoint import
            # during verification can create __pycache__ even when it is ignored
            # by the checksum manifest. Exact payload selection avoids wildcards.
            with tempfile.TemporaryDirectory(prefix='fpga_rar_payload_',dir=root.parent) as payload:
                with zipfile.ZipFile(archive) as z:z.extractall(payload)
                subprocess.run([str(rar_executable),'a','-r','-m3','-ma5','-idq',str(rar),root.name],cwd=payload,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([str(rar_executable),'t','-idq',str(rar)],stdout=log,stderr=subprocess.STDOUT,check=True)
            with tempfile.TemporaryDirectory(prefix='fpga_rar_extract_',dir=root.parent) as temp:
                subprocess.run([str(rar_executable),'x','-idq','-o-','-y',str(rar),str(Path(temp))+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
                rar_result=verify(Path(temp)/root.name)
        delivery['rar']=dict(path=str(rar.resolve()),bytes=rar.stat().st_size,sha256=sha256(rar));delivery['rar_extracted_self_check']=rar_result
    write_json(root.parent/'fpga_candidate_plus_handoff_DELIVERY.json',delivery)
    print(json.dumps({k:v for k,v in delivery.items() if k in ['version','zip','rar']},ensure_ascii=False,indent=2),flush=True)
    return delivery


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--model-root',required=True)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[3]));p.add_argument('--rar-executable')
    p.add_argument('--no-archive',action='store_true');a=p.parse_args()
    finish(a.root,a.model_root,a.repo,a.rar_executable,not a.no_archive)


if __name__=='__main__':main()
