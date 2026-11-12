"""Generate an audited, standalone Candidate-plus FPGA engineering handoff."""
import argparse
from collections import Counter
import copy
import csv
import datetime
import json
import inspect
import math
from pathlib import Path
import shutil
import sys
from xml.sax.saxutils import escape
import zipfile

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference
import onnxruntime as ort
from PIL import Image
import torch

from .dataset_adapter import letterbox, restore
from .handoff_integer import sha256, load_json, run_integer
from .handoff_ort import capture_session
from .models import RoadCandidatePlus, conv, ds
from .video_infer import sessions, overlay, banner

ROOT = Path(__file__).resolve().parents[3]
DIRS = ['01_技术文档', '02_模型与权重', '03_网络结构与算子', '04_量化参数',
        '05_硬件验证向量', '06_参考推理与转换工具', '07_性能与可视化']
MODEL_HASH = 'c7a21a9a3f1621ac5cbcbbb9030587251a340cb435413b3c086ff1c1a6d137a6'
CK_HASH = '48e6a2dccf1dc98095b41e895cd9c99f7ab044b3c8a343cb4a364275c6fceffc'
VERSION = 'CandidatePlus-PG2L200H-Handoff-v1.0.0'


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError('empty table')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def array_file(root, relative, a):
    a = np.ascontiguousarray(a)
    if a.dtype.itemsize > 1:
        a = a.astype(a.dtype.newbyteorder('<'), copy=False)
    path = root / relative;path.parent.mkdir(parents=True, exist_ok=True)
    a.tofile(path)
    np.save(path.with_suffix('.npy'), a, allow_pickle=False)
    return dict(file=Path(relative).as_posix(), dtype=a.dtype.str, shape=list(a.shape),
                bytes=a.nbytes, sha256=sha256(path), npy_file=path.with_suffix('.npy').relative_to(root).as_posix())


def trace_session(path):
    model = shape_inference.infer_shapes(onnx.load(str(path)))
    info = {v.name: v for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    del model.graph.output[:]
    names = [o for n in model.graph.node for o in n.output]
    for name in names:
        if name not in info:
            raise ValueError(f'ONNX type inference missing {name}')
        model.graph.output.append(copy.deepcopy(info[name]))
    options = ort.SessionOptions();options.intra_op_num_threads = 4;options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    runtime = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    return model, runtime, names


def parameters(value):
    # frexp convention: scale ~= multiplier / 2**right_shift, 30 significant bits.
    value = np.asarray(value, np.float64)
    mantissa, exponent = np.frexp(value)
    multiplier = np.rint(mantissa * (1 << 30)).astype(np.int64)
    shift = 30 - exponent.astype(np.int64)
    if np.any(shift < 0) or np.any(shift > 62):
        raise ValueError('unsupported fixed point exponent')
    reconstructed = np.ldexp(multiplier.astype(np.float64), -shift)
    return multiplier, shift, float(np.max(np.abs(reconstructed - value)))


def extract(root, model_root, video):
    for folder in DIRS:
        (root / folder).mkdir(parents=True, exist_ok=True)
    _, identity = sessions(model_root)
    if identity['sha256']['int8'] != MODEL_HASH or identity['sha256']['qat_checkpoint'] != CK_HASH:
        raise ValueError('Not the pinned final stage-two QAT model')
    identity.update(version=VERSION, generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    generator=str(Path(__file__).relative_to(ROOT)), source_video=str(video), source_video_sha256=sha256(video))
    write_json(root / 'MODEL_IDENTITY.json', identity)
    models = root / DIRS[1]
    copy_names = {'model.int8.qdq.onnx': model_root / 'exports/model.int8.qdq.onnx',
                  'model.fp32.onnx': model_root / 'exports/model.fp32.onnx',
                  'qat.best.pt': model_root / 'checkpoints/candidate-plus-qat/qat.best.pt',
                  'fp32.best.pt': model_root / 'checkpoints/candidate-plus/best.pt',
                  'qat.reference.torchscript.pt': model_root / 'exports/qat.reference.torchscript.pt'}
    for name, source in copy_names.items():
        shutil.copy2(source, models / name)
    tools = root / DIRS[5]
    shutil.copy2(Path(__file__).with_name('handoff_integer.py'), tools / 'handoff_integer.py')
    shutil.copy2(Path(__file__).with_name('handoff_ort.py'), tools / 'handoff_ort.py')
    provenance = tools / 'source_provenance';provenance.mkdir(exist_ok=True)
    for name in ['models.py', 'qat_stage2.py', 'quantize_stage2.py', 'export.py', 'dataset_adapter.py',
                 'build_handoff.py', 'video_infer.py', 'handoff_documentation.py', 'handoff_package.py', 'handoff_verify.py']:
        shutil.copy2(Path(__file__).with_name(name), provenance / name)
    restore_source = 'import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n'
    restore_source += '\n\n'.join(inspect.getsource(v) for v in [conv, ds, RoadCandidatePlus])
    restore_source += '''

def restore_qat(path):
    from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver, QConfig, QConfigMapping, disable_observer)
    from torch.ao.quantization.quantize_fx import prepare_qat_fx
    qconfig=QConfig(
        activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,quant_min=0,quant_max=255,
            dtype=torch.quint8,qscheme=torch.per_tensor_affine,reduce_range=False),
        weight=FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,quant_min=-128,quant_max=127,
            dtype=torch.qint8,qscheme=torch.per_channel_symmetric,ch_axis=0,reduce_range=False))
    model=prepare_qat_fx(RoadCandidatePlus().train(),QConfigMapping().set_global(qconfig),(torch.zeros(1,3,192,320),))
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval();model.apply(disable_observer)
    return model
'''
    (tools/'restore_checkpoint.py').write_text(restore_source,encoding='utf-8')
    model, runtime, names = trace_session(models / 'model.int8.qdq.onnx')
    cap = cv2.VideoCapture(str(video));ok, bgr = cap.read();cap.release()
    if not ok:
        raise ValueError('cannot decode real source video')
    x, _ = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    tensors = dict(zip(names, runtime.run(None, {'rgb': x[None].numpy()})))
    tensors['rgb'] = x[None].numpy()
    init = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    producer = {o: n for n in model.graph.node for o in n.output}
    consumers = {}
    for n in model.graph.node:
        for key in n.input:
            consumers.setdefault(key, []).append(n)
    all_rows = []
    for i, n in enumerate(model.graph.node):
        def describe(key):
            a = tensors.get(key, init.get(key))
            return dict(name=key, shape=list(a.shape), dtype=str(a.dtype), bitwidth=a.dtype.itemsize*8, bytes=a.nbytes) if a is not None else dict(name=key, optional=True)
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        attrs = {k: v.decode() if isinstance(v, bytes) else v for k, v in attrs.items()}
        ins, outs = [describe(k) for k in n.input], [describe(k) for k in n.output]
        all_rows.append(dict(index=i, name=n.name, op=n.op_type, inputs=ins, outputs=outs, attributes=attrs,
                             logical_input_bytes=sum(d.get('bytes', 0) for d in ins),
                             logical_output_bytes=sum(d.get('bytes', 0) for d in outs)))
    write_json(root / DIRS[2] / 'onnx_nodes.json', all_rows)
    write_csv(root / DIRS[2] / 'onnx_nodes.csv', all_rows)
    stages = [];q_to_id = {};channels = [];concatenated = bytearray();offsets = []
    def input_id(name):
        dq = producer[name]
        if dq.op_type != 'DequantizeLinear':
            raise ValueError('unexpected unquantized data dependency')
        return q_to_id[dq.input[0]]
    for q in model.graph.node:
        if q.op_type != 'QuantizeLinear':
            continue
        i = len(stages);sid = f's{i:02d}';data = q.input[0]
        scale, zp = float(init[q.input[1]]), int(init[q.input[2]])
        stage = dict(id=sid, qdq_tensor=q.output[0], scale=scale, zero_point=zp,
                     dtype='uint8', shape=list(tensors[q.output[0]].shape), inputs=[], macs=0)
        q_to_id[q.output[0]] = sid
        if data == 'rgb':
            stage.update(op='Input', name='RGB input quantizer')
        else:
            prev = producer[data];relu = prev.op_type == 'Relu'
            if relu:
                prev = producer[prev.input[0]]
            stage.update(op=prev.op_type, name=prev.name, inputs=[input_id(prev.input[0])])
            if prev.op_type == 'Conv':
                attrs = {a.name: helper.get_attribute_value(a) for a in prev.attribute}
                wdq = producer[prev.input[1]]
                w, sw, zw = [init[k] for k in wdq.input]
                if w.dtype != np.int8 or np.any(zw != 0) or sw.shape != (w.shape[0],):
                    raise ValueError('unsupported weight quantization')
                source = stages[int(stage['inputs'][0][1:])]
                sx = source['scale'];bfloat = init[prev.input[2]]
                bias_scale = sx * sw.astype(np.float64)
                bias64 = np.rint(bfloat.astype(np.float64) / bias_scale).astype(np.int64)
                if np.any(abs(bias64) >= (1 << 31)):
                    raise ValueError('bias exceeds INT32')
                bias = bias64.astype('<i4');ratio = bias_scale / scale
                m, shifts, err = parameters(ratio)
                base = f'{DIRS[1]}/standard_weights/{sid}'
                ws = array_file(root, base + '.weight.bin', w)
                bs = array_file(root, f'{DIRS[3]}/{sid}.bias_int32.bin', bias)
                fs = array_file(root, f'{DIRS[3]}/{sid}.bias_float32.bin', bfloat)
                ss = array_file(root, f'{DIRS[3]}/{sid}.weight_scale.bin', sw)
                zs = array_file(root, f'{DIRS[3]}/{sid}.weight_zero_point.bin', zw)
                array_file(root, f'{DIRS[3]}/{sid}.multiplier.bin', m.astype('<i4'))
                array_file(root, f'{DIRS[3]}/{sid}.right_shift.bin', shifts.astype('<i4'))
                k = int(np.prod(w.shape[1:]));macs = int(np.prod(stage['shape'])) * k
                max_x = max(source['zero_point'], 255-source['zero_point'])
                bounds = max_x * np.abs(w.astype(np.int64)).sum((1, 2, 3)) + np.abs(bias64)
                if np.any(bounds >= (1 << 31)) or np.any(bounds > np.iinfo(np.int64).max // m):
                    raise OverflowError('static accumulation or product bound exceeds contract width')
                stage.update(weights=ws, bias_int32=bs, original_bias_float32=fs,
                    weight_scale=ss, weight_zero_point=zs, input_zero_point=source['zero_point'],
                    input_scale=sx, weight_scales=sw.tolist(), weight_zero_points=zw.tolist(), bias_scales=bias_scale.tolist(),
                    bias_int32_values=bias.tolist(), multipliers=m.tolist(), right_shifts=shifts.tolist(),
                    multiplier_max_abs_error=err, bias_rounding_max_abs_real_error=float(np.max(np.abs(bias64*bias_scale-bfloat))),
                    kernel=list(w.shape[-2:]), strides=attrs['strides'], pads=attrs['pads'],
                    dilations=attrs['dilations'], groups=attrs['group'], relu=relu,
                    macs=macs, weight_parameters=w.size, bias_parameters=bias.size,
                    accumulator_bound=int(bounds.max()), minimum_safe_signed_accumulator_bits=int(bounds.max()).bit_length()+1,
                    requant_product_bound=int(np.max(bounds*m)))
                for c in range(len(sw)):
                    channels.append(dict(stage=sid, onnx_name=prev.name, output_channel=c, input_scale=sx,
                        input_zero_point=source['zero_point'], weight_scale=float(sw[c]), weight_zero_point=int(zw[c]),
                        output_scale=scale, output_zero_point=zp, original_bias_float32=float(bfloat[c]),
                        bias_int32=int(bias[c]), bias_scale=float(bias_scale[c]), multiplier=int(m[c]),
                        right_shift=int(shifts[c]), effective_ratio=float(ratio[c]), accumulator_bound=int(bounds[c])))
                offsets.append(dict(stage=sid, offset_bytes=len(concatenated), length_bytes=w.nbytes, shape=list(w.shape), layout='OIHW'))
                concatenated.extend(w.tobytes())
            elif prev.op_type == 'Resize':
                attrs = {a.name: helper.get_attribute_value(a) for a in prev.attribute}
                if attrs['mode'] != b'nearest' or attrs['coordinate_transformation_mode'] != b'asymmetric' or attrs['nearest_mode'] != b'floor':
                    raise ValueError('unsupported Resize')
                source = stages[int(stage['inputs'][0][1:])]
                if scale != source['scale'] or zp != source['zero_point']:
                    raise ValueError('Resize changes quantization domain')
                stage.update(mode='nearest', coordinate_transformation_mode='asymmetric', nearest_mode='floor')
            elif prev.op_type == 'Add':
                stage['inputs'].append(input_id(prev.input[1]))
                sources = [stages[int(k[1:])] for k in stage['inputs']]
                ratios = [s['scale']/scale for s in sources]
                multipliers = [int(round(v*(1 << 31))) for v in ratios]
                if max(multipliers) >= (1 << 31):
                    raise ValueError('Add multiplier requires wider format')
                stage.update(input_scales=[s['scale'] for s in sources],
                             input_zero_points=[s['zero_point'] for s in sources], multipliers=multipliers,
                             right_shift=31, multiplier_ratios=ratios,
                             multiplier_max_abs_error=max(abs(m/(1 << 31)-r) for m, r in zip(multipliers, ratios)))
            else:
                raise ValueError(f'unsupported data op {prev.op_type}')
        stage['output_bytes'] = int(np.prod(stage['shape']))
        stage['input_bytes'] = sum(stages[int(k[1:])]['output_bytes'] for k in stage['inputs'])
        stage['logical_io_bytes'] = stage['input_bytes']+stage['output_bytes']+stage.get('weights', {}).get('bytes', 0)+stage.get('bias_int32', {}).get('bytes', 0)
        stage['output_bitwidth'] = 8;stage['accumulator_bitwidth'] = 32 if stage['op'] == 'Conv' else None
        stages.append(stage)
    if len(stages) != 20 or sum(s['macs'] for s in stages) != 84718080:
        raise ValueError('architecture does not match pinned Candidate-plus')
    (models / 'weights.oihw.int8.bin').write_bytes(concatenated)
    write_json(models / 'weights.offsets.json', dict(endianness='little', layout='OIHW C contiguous; no alignment/padding', segments=offsets,
               bytes=len(concatenated), sha256=sha256(models/'weights.oihw.int8.bin')))
    # Integer input LUT compiled from the exact software FP32 /255 path.
    lut = np.clip(np.rint((np.arange(256, dtype=np.float32)/np.float32(255))/np.float32(stages[0]['scale']))+stages[0]['zero_point'], 0, 255).astype(np.uint8)
    lut_spec = array_file(root, f'{DIRS[3]}/input_rgb_byte_to_quantized_lut.bin', lut)
    contract = dict(version='candidate-plus-integer-v1', status='derived integer contract, not FPGA measured or QDQ bit-equivalence',
        model_sha256=MODEL_HASH, activation='uint8 per-tensor affine', weight='int8 per-output-channel axis0 symmetric zero_point0',
        accumulators='signed int32; overflow is an error, no intermediate wrapping',
        requantization='signed int64 product, power-of-two right shift RNE ties-even, output_zero_point then saturate [0,255]',
        bias='RNE(float64(bias_float32)/(float64(input_scale)*float64(weight_scale))) -> int32',
        input_lut=lut_spec, stages=stages)
    write_json(root / DIRS[3] / 'integer_contract.json', contract)
    write_json(root / DIRS[2] / 'integer_stages.json', stages)
    simple = [{k: s.get(k) for k in ['id','name','op','inputs','shape','kernel','strides','pads','groups','relu','weight_parameters',
              'bias_parameters','macs','input_bytes','output_bytes','logical_io_bytes','output_bitwidth','accumulator_bitwidth']} for s in stages]
    write_csv(root / DIRS[2] / 'integer_stages.csv', simple)
    write_csv(root / DIRS[3] / 'per_output_channel_quantization.csv', channels)
    write_json(root / DIRS[3] / 'activation_quantization.json', [{k:s[k] for k in ['id','qdq_tensor','op','shape','scale','zero_point','dtype']} for s in stages])
    remaining = Counter(k for s in stages for k in s['inputs']);live={};peak=0;peak_with_acc=0;timeline=[]
    for s in stages:
        live[s['id']]=s['output_bytes'];size=sum(live.values());peak=max(peak,size)
        extra = s['output_bytes']*4 if s['op']=='Conv' else 0
        peak_with_acc=max(peak_with_acc,size+extra)
        timeline.append(dict(stage=s['id'],live_tensors=list(live),live_feature_bytes=size,full_int32_accumulator_bytes=extra))
        for k in s['inputs']:
            remaining[k]-=1
            if remaining[k]==0:live.pop(k)
    training=RoadCandidatePlus();train_params=sum(v.numel() for v in training.parameters())
    summary=dict(version=VERSION, training_parameters=train_params, conv_layers=14,
        depthwise_layers=sum(s.get('groups',1)>1 for s in stages), pointwise_layers=sum(s.get('kernel')==[1,1] for s in stages),
        onnx_operators=dict(Counter(n.op_type for n in model.graph.node)), onnx_nodes=len(model.graph.node),
        macs=sum(s['macs'] for s in stages), folded_weight_elements=len(concatenated),
        folded_bias_elements=sum(s.get('bias_parameters',0) for s in stages),
        folded_bias_int32_bytes=sum(s.get('bias_int32',{}).get('bytes',0) for s in stages),
        raw_weight_blob_bytes=len(concatenated), qdq_onnx_bytes=(models/'model.int8.qdq.onnx').stat().st_size,
        fp32_onnx_bytes=(models/'model.fp32.onnx').stat().st_size,
        peak_live_uint8_features_bytes=peak, peak_features_plus_full_int32_accumulator_bytes=peak_with_acc,
        activation_memory_timeline=timeline, logical_unfused_io_bytes_per_frame=sum(s['logical_io_bytes'] for s in stages),
        memory_scope='Batch1 logical tensors, distinct buffers, output allocated before freeing consumed inputs; excludes alignment, DMA, banks, line buffers, implementation caches; full accumulator optional in RTL',
        largest_uint8_feature_bytes=max(s['output_bytes'] for s in stages),
        checkpoint_parameter_explanation='32,524 trainable parameters include 11 BN gamma/beta; final QAT Conv has BN folded into weights and bias',
        runtime_versions=dict(python=sys.version,numpy=np.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__,torch=torch.__version__,opencv=cv2.__version__))
    if train_params != 32524:
        raise ValueError('training parameter count changed')
    write_json(root / DIRS[2] / 'architecture_summary.json', summary)
    diagram=['flowchart TD']
    for s in stages:
        shape='×'.join(map(str,s['shape'][1:]))
        label=f"{s['id']} {s['op']} {s.get('kernel','')} {'+ ReLU' if s.get('relu') else ''}<br/>{shape} UINT8"
        diagram.append(f'  {s["id"]}["{label}"]')
        for key in s['inputs']:diagram.append(f'  {key} --> {s["id"]}')
    (root / DIRS[2] / 'network.mmd').write_text('\n'.join(diagram)+'\n',encoding='utf-8')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1840" viewBox="0 0 1000 1840">',
         '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#253d55"/></marker></defs>',
         '<rect width="1000" height="1840" fill="white"/>',
         '<text x="30" y="32" font-family="sans-serif" font-size="22">Candidate-plus final QAT: 14 Conv / 84,718,080 MACs</text>']
    for i,s in enumerate(stages):
        y=60+i*86;fill={'Conv':'#e4effb','Resize':'#fff3cd','Add':'#e3f3e6','Input':'#eee'}[s['op']]
        for k in s['inputs']:
            j=int(k[1:]);py=60+j*86
            if j==i-1:svg.append(f'<path d="M420,{py+64} L420,{y}" stroke="#253d55" stroke-width="2" marker-end="url(#arrow)"/>')
            else:
                route=760 if j==1 else 850
                svg.append(f'<path d="M700,{py+32} L{route},{py+32} L{route},{y+32} L700,{y+32}" fill="none" stroke="#253d55" stroke-width="2" marker-end="url(#arrow)"/>')
                svg.append(f'<text x="{route+6}" y="{(py+y)//2+32}" font-family="sans-serif" font-size="14">{k} skip</text>')
        label=f"{s['id']}  {s['op']} {s.get('kernel','')} {'+ ReLU' if s.get('relu') else ''}"
        shape=f"NCHW {s['shape']} | U8 s={s['scale']:.8g} z={s['zero_point']}"
        svg.extend([f'<rect x="140" y="{y}" width="560" height="64" rx="8" fill="{fill}" stroke="#253d55"/>',
                    f'<text x="156" y="{y+25}" font-family="monospace" font-size="17">{escape(label)}</text>',
                    f'<text x="156" y="{y+49}" font-family="monospace" font-size="14">{escape(shape)}</text>'])
    svg.append('</svg>');(root/DIRS[2]/'network.svg').write_text('\n'.join(svg),encoding='utf-8')
    print('extracted',json.dumps({k:v for k,v in summary.items() if not isinstance(v,(dict,list))},ensure_ascii=False),flush=True)
    return contract, summary


def vectors(root, video, contract):
    """Six genuine, distinct scenes from the supplied clip, not dataset GT."""
    capture,optimized=capture_session(root,contract,root/DIRS[1]/'ort_cpu.optimized.onnx')
    write_json(root/DIRS[2]/'ort_cpu_fused_graph.json',dict(operators=dict(Counter(n.op_type for n in optimized.graph.node)),
        note='CPU-specific ORT fusion evidence, not the portable source graph or an FPGA model',
        nodes=[dict(name=n.name,domain=n.domain,op=n.op_type,inputs=list(n.input),outputs=list(n.output)) for n in optimized.graph.node]))
    choices=[(0,'crosswalk_front','近处真实斑马线，覆盖存在缺口'),
             (245,'dashed_white_car','正常直行、虚线和白车尾部误检'),
             (612,'solid_arrows','实线与道路箭头，箭头发生斑马线误检'),
             (857,'normal_bus_truck','正常道路、公交车与货车遮挡'),
             (1225,'crosswalk_turn','转弯视角的横向斑马线'),
             (1277,'hard_glare_background','转弯、玻璃反射和背景非道路误响应')]
    rows=[];all_layer=[];lut=np.fromfile(root/contract['input_lut']['file'],np.uint8)
    for number,(frame,tag,description) in enumerate(choices):
        folder=root/DIRS[4]/f'{number:03d}_{tag}';folder.mkdir(parents=True,exist_ok=True)
        cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,bgr=cap.read();position=int(cap.get(cv2.CAP_PROP_POS_FRAMES));cap.release()
        if not ok or position != frame+1:
            raise RuntimeError('failed exact source frame seek')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);Image.fromarray(rgb).save(folder/'source.png')
        x,g=letterbox(rgb);x=x[None].numpy();native=(x*np.float32(255)).round().astype(np.uint8)
        qin=lut[native]
        expected=np.clip(np.rint(x/np.float32(contract['stages'][0]['scale']))+contract['stages'][0]['zero_point'],0,255).astype(np.uint8)
        np.testing.assert_array_equal(qin,expected)
        logits,qdq=capture(x)
        integer,values,acc=run_integer(root,qin)
        arrays={}
        def put(name,array,scale=None,zp=None,semantics=None):
            relative=folder.relative_to(root).as_posix()+'/'+name+'.bin'
            spec=array_file(root,relative,array)
            spec.update(scale=scale,zero_point=zp,semantics=semantics)
            arrays[name]=spec
            return spec
        put('input_rgb_u8',native,1/255,0,'letterboxed RGB bytes NCHW; not yet activation-quantized')
        put('input_fp32',x,1.0,0,'RGB /255 ONNX input')
        put('input_quantized_u8',qin,contract['stages'][0]['scale'],contract['stages'][0]['zero_point'],'integer core input')
        last=contract['stages'][-1]
        iqlogits=(integer.astype(np.float32)-last['zero_point'])*np.float32(last['scale'])
        put('output_qdq_logits_fp32',logits,1.0,0,'actual final QAT ONNX output')
        put('output_qdq_uint8',qdq[last['id']],last['scale'],last['zero_point'],'actual final ORT fused QDQ quantizer values')
        put('output_integer_uint8',integer,last['scale'],last['zero_point'],'derived integer-v1 output, not guaranteed QDQ exact')
        put('output_integer_logits_fp32',iqlogits,1.0,0,'host dequantization of integer-v1 output')
        sample_layers=[]
        for s in contract['stages']:
            ref=qdq[s['id']];actual=values[s['id']]
            put('qdq/'+s['id'],ref,s['scale'],s['zero_point'],'ORT actual quantizer output')
            put('integer/'+s['id'],actual,s['scale'],s['zero_point'],'integer-v1 stage output')
            delta=actual.astype(np.int16)-ref.astype(np.int16)
            layer=dict(sample=number,frame=frame,stage=s['id'],shape=s['shape'],
                       max_abs_lsb=int(abs(delta).max()),mean_abs_lsb=float(abs(delta).mean()),mismatch_count=int(np.count_nonzero(delta)),elements=delta.size)
            sample_layers.append(layer);all_layer.append(layer)
            if s['id'] in acc:
                put('accumulator/'+s['id'],acc[s['id']],s['bias_scales'],0,'exact INT32 sum((qx-zx)*qw)+b32 before ReLU/requant')
        for label,output in [('qdq',logits),('integer',iqlogits)]:
            # Host visualization follows training/video geometry, outside integer core.
            probability=torch.from_numpy(output).sigmoid().numpy()[0]
            masks=probability>=.5
            put(label+'_mask_320x192',masks.astype(np.uint8),None,None,'four independent binary masks, padding retained here')
            restored=restore(probability,g)>=.5
            for c,name in enumerate(['drivable_area','solid_lane','dashed_lane','crosswalk']):
                Image.fromarray((restored[c]*255).astype(np.uint8)).save(folder/f'{label}_{name}_native.png')
            vis=overlay(rgb,restored);head=banner(rgb.shape[1],f'{label.upper()} | {tag} | {frame/30:.3f}s')
            vis[:len(head)]=head;Image.fromarray(vis).save(folder/f'{label}_overlay.jpg',quality=93)
        mask_agreement=float(((integer>=last['zero_point'])==(qdq[last['id']]>=last['zero_point'])).mean())
        metadata=dict(sample=number,id=folder.name,description=description,source_video=str(video),source_video_sha256=sha256(video),
            frame_zero_based=frame,time_seconds=frame/30,geometry=g,ground_truth='none: real video frame, not supervised accuracy evaluation',
            arrays=arrays,qdq_vs_integer_layers=sample_layers,model_grid_mask_agreement=mask_agreement,
            qdq_vs_integer_max_logit_error=float(abs(iqlogits-logits).max()),
            qdq_vs_integer_mean_abs_logit_error=float(abs(iqlogits-logits).mean()))
        write_json(folder/'sample.json',metadata)
        rows.append({k:v for k,v in metadata.items() if k not in ['arrays','qdq_vs_integer_layers']})
        print(f'vector {number}: frame={frame} mask_agreement={mask_agreement:.8f} max_logit_error={metadata["qdq_vs_integer_max_logit_error"]:.9f}',flush=True)
    write_json(root/DIRS[4]/'samples.json',rows)
    write_csv(root/DIRS[4]/'qdq_vs_integer_per_layer.csv',all_layer)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=str(ROOT/'outputs/fpga_candidate_plus_handoff'))
    p.add_argument('--model-root',default=str(ROOT/'outputs/fpga_road_seg_stage2_qat'))
    p.add_argument('--video',default='D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/video_20260929_112507.mp4')
    a=p.parse_args();out=Path(a.output).resolve()
    contract,summary=extract(out,Path(a.model_root).resolve(),Path(a.video).resolve())
    vectors(out,Path(a.video).resolve(),contract)


if __name__=='__main__':main()
