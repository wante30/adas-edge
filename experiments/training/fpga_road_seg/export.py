"""Fixed-shape FP32 ONNX + Conv W8A8 static PTQ, with executable comparisons.

Official API: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
QDQ is an interchange graph, NOT a PG2L200H memory/RTL weight image.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
import torch

from .dataset_adapter import RoadDataset, load_manifest, letterbox, file_hash, write_json
from .engine import Metrics
from .train import load_checkpoint, names_for


def session(path):
    options=ort.SessionOptions(); options.intra_op_num_threads=4
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])


def inventory(path):
    graph=shape_inference.infer_shapes(onnx.load(str(path))).graph
    shapes={}
    for value in (*graph.input,*graph.value_info,*graph.output):
        shapes[value.name]=[d.dim_value if d.HasField('dim_value') else d.dim_param for d in value.type.tensor_type.shape.dim]
    initializers={i.name:numpy_helper.to_array(i) for i in graph.initializer}
    nodes=[]
    for node in graph.node:
        attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
        # Retain scalar/list attributes; tensor attributes are listed by dimensions.
        attrs={k:dict(tensor_dims=list(v.dims)) if isinstance(v,onnx.TensorProto) else v.decode() if isinstance(v,bytes) else v for k,v in attrs.items()}
        nodes.append(dict(name=node.name,op=node.op_type,inputs=list(node.input),outputs=list(node.output),
                          output_shapes=[shapes.get(n) for n in node.output],attributes=attrs))
    return dict(operators=dict(Counter(n['op'] for n in nodes)),nodes=nodes,
                initializers=[dict(name=n,shape=list(a.shape),dtype=str(a.dtype)) for n,a in initializers.items()])


class ManifestReader(CalibrationDataReader):
    def __init__(self, rows, size):
        self.rows=list(rows); self.size=size; self.index=0

    def get_next(self):
        if self.index>=len(self.rows):
            return None
        row=self.rows[self.index]; self.index+=1
        with Image.open(row['image']) as im:
            x,_=letterbox(np.asarray(im.convert('RGB')),size=self.size)
        return {'rgb':x[None].numpy()}


def multiplier_shift(value):
    """Approximate positive multiplier by integer*2^-shift; not a vendor format."""
    a=np.asarray(value,dtype=np.float64)
    mantissa,exponent=np.frexp(a)
    integer=np.rint(mantissa*(1<<30)).astype(np.int64)
    shift=30-exponent
    approx=integer.astype(np.float64)*np.exp2(-shift.astype(np.float64))
    return dict(multiplier=integer.tolist(),right_shift=shift.tolist(),
                max_absolute_error=float(np.max(np.abs(approx-a))))


def quantization_metadata(path):
    model=onnx.load(str(path)); init={i.name:numpy_helper.to_array(i) for i in model.graph.initializer}
    # QAT exporter aliases shared scalar constants through Identity nodes.
    pending=list(model.graph.node)
    for _ in range(len(pending)):
        changed=False
        for node in pending:
            if node.output[0] in init:continue
            if node.op_type=='Identity' and node.input[0] in init:
                init[node.output[0]]=init[node.input[0]];changed=True
            elif node.op_type=='Constant':
                tensor=next((a.t for a in node.attribute if a.name=='value'),None)
                if tensor is not None:init[node.output[0]]=numpy_helper.to_array(tensor);changed=True
        if not changed:break
    producers={o:n for n in model.graph.node for o in n.output}
    consumers={}
    for n in model.graph.node:
        for i in n.input:
            consumers.setdefault(i,[]).append(n)
    records=[]
    def describe(name):
        node=producers.get(name)
        if node is None or node.op_type!='DequantizeLinear':
            return None
        attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
        scale=init[node.input[1]]
        zp=init[node.input[2]] if len(node.input)>2 else np.array(0,np.uint8)
        data=init.get(node.input[0])
        return dict(quantized_tensor=node.input[0],scale=scale.tolist(),zero_point=zp.tolist(),
                    axis=attrs.get('axis'),storage_dtype=str(zp.dtype),
                    shape=list(data.shape) if data is not None else None)
    for node in model.graph.node:
        if node.op_type!='Conv':
            continue
        activation=describe(node.input[0]); weight=describe(node.input[1])
        bias=describe(node.input[2]) if len(node.input)>2 else None
        outputs=[n for n in consumers.get(node.output[0],[]) if n.op_type=='QuantizeLinear']
        output=None
        if outputs:
            q=outputs[0]
            output=dict(scale=init[q.input[1]].tolist(),zero_point=init[q.input[2]].tolist(),storage_dtype=str(init[q.input[2]].dtype))
        rec=dict(name=node.name,activation=activation,weight=weight,bias=bias,output=output)
        if len(node.input)>2 and bias is None:
            raw_bias=init.get(node.input[2])
            rec['bias_storage']=str(raw_bias.dtype) if raw_bias is not None else 'unresolved'
            rec['bias_contract']='FP32 bias in reference graph; conversion to int32 input_scale*weight_scale remains a hardware adaptation'
        if activation and weight and output:
            s=np.asarray(activation['scale'])*np.asarray(weight['scale'])/np.asarray(output['scale'])
            rec['requantization']=dict(real_multiplier=s.tolist(),**multiplier_shift(s))
            if bias:
                rec['bias_contract']='int32 bias with scale=input_scale*weight_scale, zero_point=0; accumulator/rounding/saturation require RTL confirmation'
        records.append(rec)
    return dict(format='ONNX QDQ Conv quantization',weight_bits=8,conv_activation_bits=8,
                residual_resize_io='float32 outside Conv QDQ boundaries; not a fully integer graph',
                shift_contract='illustrative Q30 multiplier/right shift approximation; not validated on FPGA',layers=records)


def compare(model, fp32_path, int8_path, manifest, cfg, split='val', disabled_channels=(), compare_count=0):
    fp,iq=session(fp32_path),session(int8_path)
    names=names_for(cfg['model']); ds=RoadDataset(manifest,split,(cfg['width'],cfg['height']),len(names)==2,disabled_channels)
    if compare_count:
        ds.rows = ds.rows[:compare_count]
    fm,im=Metrics(len(names)),Metrics(len(names))
    torch_error=int8_error=0.; probability_error=[]; agreement=total=0
    model.eval()
    threshold=cfg.get('threshold',.5)
    for x,y,v in ds:
        inputs=x[None].numpy(); a=fp.run(None,{'rgb':inputs})[0]; b=iq.run(None,{'rgb':inputs})[0]
        with torch.no_grad():
            ref=model(x[None]).numpy()
        torch_error=max(torch_error,float(np.max(np.abs(a-ref))))
        int8_error=max(int8_error,float(np.max(np.abs(a-b))))
        pa,pb=torch.from_numpy(a).sigmoid(),torch.from_numpy(b).sigmoid()
        fm.update(pa,y[None],v[None],threshold); im.update(pb,y[None],v[None],threshold)
        probability_error.append(float(torch.mean(torch.abs(pa-pb))))
        active=v[None]>.5
        agreement+=int((((pa>=threshold)==(pb>=threshold)) & active).sum()); total+=int(active.sum())
    if torch_error>1e-3:
        raise RuntimeError(f'FP32 ONNX parity exceeds 1e-3: {torch_error}')
    return dict(split=split,samples=len(ds),torch_onnx_max_logit_error=torch_error,
                fp32_int8_max_logit_error=int8_error,mean_probability_absolute_error=float(np.mean(probability_error)),
                threshold_mask_agreement_valid_pixels=agreement/total if total else None,
                fp32=fm.report(names),int8=im.report(names),threshold=threshold,
                note='Synthetic data metrics are software smoke only; real quantization acceptance requires labelled held-out road data.')


def export(checkpoint, manifest, output, calibration_count=16, compare_split='val', synthetic_smoke=False, compare_count=0):
    torch.set_num_threads(4)
    model,ck=load_checkpoint(checkpoint)
    cfg=ck['config']; size=cfg['width'],cfg['height']
    all_rows=load_manifest(manifest)
    if ck['evidence']=='synthetic_smoke' and not synthetic_smoke:
        raise ValueError('synthetic-trained checkpoint requires --synthetic-smoke')
    if any(r['source']=='synthetic' for r in all_rows) and not synthetic_smoke:
        raise ValueError('synthetic calibration requires --synthetic-smoke')
    candidates=sorted([r for r in all_rows if r['split']=='train'],key=lambda r:r['id'])
    indices=np.linspace(0,len(candidates)-1,min(calibration_count,len(candidates)),dtype=int) if candidates else []
    rows=[candidates[int(i)] for i in indices]
    if calibration_count<1 or not rows:
        raise ValueError('nonempty TRAIN-only PTQ calibration required')
    # load_manifest validates global sequence/content leakage before selecting calibration.
    out=Path(output)
    completed=out/'export-report.json'
    if completed.exists():
        previous=__import__('json').loads(completed.read_text(encoding='utf-8'))
        if previous.get('checkpoint_sha256')==file_hash(checkpoint) and previous.get('dataset_manifest_sha256')==file_hash(manifest):
            return previous
        raise ValueError('completed export belongs to a different checkpoint/manifest; use a new output')
    if out.exists():
        backup=out.with_name(out.name + f'.incomplete-{time.time_ns()}')
        out.rename(backup)
    out.mkdir(parents=True,exist_ok=False)
    fp=out/'model.fp32.onnx'; prepared=out/'model.preprocessed.onnx'; iq=out/'model.int8.qdq.onnx'
    model.eval()
    x=torch.zeros(1,3,size[1],size[0])
    with torch.no_grad():
        torch.onnx.export(model,x,str(fp),input_names=['rgb'],output_names=['logits'],opset_version=17,
                          do_constant_folding=True,dynamic_axes=None)
    onnx.checker.check_model(onnx.load(str(fp)))
    quant_pre_process(str(fp),str(prepared),skip_symbolic_shape=True)
    reader=ManifestReader(rows,size)
    quantize_static(str(prepared),str(iq),reader,quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,
                    op_types_to_quantize=['Conv'],extra_options={'ActivationSymmetric':True,'WeightSymmetric':True})
    onnx.checker.check_model(onnx.load(str(iq)))
    fp_inventory,iq_inventory=inventory(fp),inventory(iq)
    quant=quantization_metadata(iq)
    if not quant['layers'] or any(r['weight'] is None or r['activation'] is None for r in quant['layers']):
        raise RuntimeError('not every Conv has weight/activation quantization metadata')
    write_json(out/'fp32-operators.json',fp_inventory); write_json(out/'int8-operators.json',iq_inventory)
    write_json(out/'quantization.json',quant)
    calibration=[dict(id=r['id'],group_id=r['group_id'],split=r['split'],image=r['image'],sha256=file_hash(r['image']),source=r['source'],license=r['license']) for r in rows]
    write_json(out/'calibration-manifest.json',calibration)
    comparison=compare(model,fp,iq,manifest,cfg,compare_split,ck.get('excluded_channels', ()),compare_count)
    write_json(out/'comparison.json',comparison)
    allowed={'Conv','Relu','Add','Resize','Constant','Identity','Shape','Gather','Slice','Concat','Unsqueeze'}
    unsupported=sorted(set(fp_inventory['operators'])-allowed)
    report=dict(evidence=ck['evidence'],model=cfg['model'],checkpoint_sha256=file_hash(checkpoint),
                excluded_channels=ck.get('excluded_channels', []),
                deployment_note='Excluded output channels are untrained/invalid and must be suppressed by consumers, even though the fixed graph retains their logits.',
                dataset_manifest_sha256=file_hash(manifest),ort_version=ort.__version__,onnx_version=onnx.__version__,
                input=dict(name='rgb',shape=[1,3,size[1],size[0]],dtype='float32',color='RGB',normalization='uint8 / 255; letterbox fill=114'),
                output=dict(name='logits',shape=[1,len(names_for(cfg['model'])),size[1],size[0]],class_names=names_for(cfg['model']),sigmoid='software; binary channel threshold 0.5'),
                fp32_sha256=file_hash(fp),int8_sha256=file_hash(iq),calibration_samples=len(rows),
                calibration_selection='uniformly spaced over sorted TRAIN IDs; no validation/test calibration',
                comparison=comparison,fp32_operators=fp_inventory['operators'],int8_operators=iq_inventory['operators'],
                hypothetical_profile='Conv/Depthwise/Pointwise/ReLU/Add/nearest Resize; NOT confirmed FPGA capabilities',
                operators_outside_candidate_profile=unsupported,hardware_status='awaiting teammate; no board measurements',
                integer_scope=quant['residual_resize_io'])
    write_json(out/'export-report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True); p.add_argument('--calibration-count',type=int,default=16)
    p.add_argument('--compare-split',choices=['val','test'],default='val')
    p.add_argument('--synthetic-smoke',action='store_true')
    p.add_argument('--compare-count',type=int,default=0,help='explicit pilot comparison subset; zero evaluates the full split')
    args=p.parse_args()
    if args.compare_count < 0:
        p.error('--compare-count must be nonnegative')
    print(export(args.checkpoint,args.manifest,args.output,args.calibration_count,args.compare_split,args.synthetic_smoke,args.compare_count)['comparison'])


if __name__=='__main__':
    main()
