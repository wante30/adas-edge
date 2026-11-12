"""Bounded real-data QAT fallback after FP32 optimization and measured PTQ loss."""
import copy
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver, QConfig, QConfigMapping,
    disable_fake_quant, disable_observer, enable_fake_quant)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_to_reference_fx
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import CLASSES
from .dataset_adapter import file_hash, read_json, write_json
from .engine import evaluate, masked_loss
from .train import AugmentedRoadDataset, load_checkpoint


class ExportQuantizedConv(torch.nn.Module):
    """Reference folded Conv represented with standard exportable fake-quant ops."""
    def __init__(self, conv):
        super().__init__()
        self.register_buffer('weight',conv.weight.detach().clone())
        self.register_buffer('bias',conv.bias.detach().clone() if conv.bias is not None else None)
        self.register_buffer('scale',conv.weight_scale.detach().clone())
        self.register_buffer('zero_point',conv.weight_zero_point.detach().clone().to(torch.int32))
        self.stride,self.padding,self.dilation,self.groups=conv.stride,conv.padding,conv.dilation,conv.groups

    def forward(self,x):
        weight=torch.fake_quantize_per_channel_affine(self.weight,self.scale,self.zero_point,0,-128,127)
        return F.conv2d(x,weight,self.bias,self.stride,self.padding,self.dilation,self.groups)


def exportable_reference(prepared):
    reference=convert_to_reference_fx(copy.deepcopy(prepared).cpu().eval())
    from torch.ao.nn.quantized.reference import Conv2d
    def replace(parent):
        for name,child in list(parent.named_children()):
            if isinstance(child,Conv2d):setattr(parent,name,ExportQuantizedConv(child))
            else:replace(child)
    replace(reference)
    for node in list(reference.graph.nodes):
        if node.op=='call_function' and node.target==torch.quantize_per_tensor:
            node.target=torch.fake_quantize_per_tensor_affine
            node.args=(*node.args[:3],0,255)
        elif node.op=='call_method' and node.target=='dequantize':
            node.replace_all_uses_with(node.args[0]);reference.graph.erase_node(node)
    reference.graph.lint();reference.recompile()
    return reference


def run_qat(checkpoint, manifest, output, calibration, validation):
    from .quantize_stage2 import evaluate_onnx, domain_score
    from .export import session, inventory
    out=Path(output);report_file=out/'qat.json'
    if report_file.exists():return read_json(report_file)
    random.seed(151);np.random.seed(151);torch.manual_seed(151)
    teacher,ck=load_checkpoint(checkpoint,'cuda');teacher.eval()
    qconfig=QConfig(
        activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,quant_min=0,quant_max=255,
            dtype=torch.quint8,qscheme=torch.per_tensor_affine,reduce_range=False),
        weight=FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,quant_min=-128,quant_max=127,
            dtype=torch.qint8,qscheme=torch.per_channel_symmetric,ch_axis=0,reduce_range=False))
    model=prepare_qat_fx(copy.deepcopy(teacher).cpu().train(),QConfigMapping().set_global(qconfig),
                         (torch.zeros(1,3,192,320),)).cuda()
    statefile=out/'qat.last.pt'
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    history=[];start=0;best_score=-float('inf')
    if statefile.exists():
        state=torch.load(statefile,map_location='cuda',weights_only=False)
        if state['fp32_checkpoint_sha256']!=file_hash(checkpoint):raise ValueError('QAT checkpoint identity mismatch')
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];start=state['epoch'];best_score=state['best_score']
    else:
        # Ranges from training images only. Freeze observers before validation;
        # validation cannot update a calibration range or batch-normalization.
        from .export import ManifestReader
        model.eval();model.apply(disable_fake_quant)
        reader=ManifestReader(calibration,(320,192))
        with torch.no_grad():
            while True:
                item=reader.get_next()
                if item is None:break
                model(torch.from_numpy(item['rgb']).cuda())
    model.apply(enable_fake_quant);model.apply(disable_observer)
    train=AugmentedRoadDataset(manifest,'train')
    counts=Counter(r['source'] for r in train.rows)
    weights=[(.8 if r['source']=='bdd100k' else .2)/counts[r['source']] for r in train.rows]
    sampler=WeightedRandomSampler(weights,16000,replacement=True)
    loader=DataLoader(train,batch_size=16,sampler=sampler,num_workers=4,persistent_workers=True,pin_memory=True)
    vals={s:DataLoader(ds,batch_size=32,num_workers=2,persistent_workers=True,pin_memory=True) for s,ds in validation.items()}
    for epoch in range(start,3):
        model.train();model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        total=n=0;t0=time.perf_counter()
        for x,y,v in loader:
            x,y,v=x.cuda(),y.cuda(),v.cuda();optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():reference=teacher(x)
            logits=model(x)
            supervised=masked_loss(logits,y,v,ck['config']['pos_weight'])
            # FP32 logits regularize the quantized model, not new ground-truth labels.
            loss=supervised+.2*F.mse_loss(logits,reference)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite QAT loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(x);n+=len(x)
        domains={s:evaluate(model,dl,'cuda',CLASSES) for s,dl in vals.items()}
        score=domain_score(domains);improved=score>best_score
        best_score=max(score,best_score)
        history.append(dict(epoch=epoch+1,loss=total/n,seconds=time.perf_counter()-t0,domains=domains,score=score))
        state=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history,
                   best_score=best_score,fp32_checkpoint_sha256=file_hash(checkpoint),
                   manifest_sha256=file_hash(manifest),method='QAT fake quant W8A8 with frozen BN/ranges and FP32-logit distillation regularizer')
        torch.save(state,out/'qat.last.tmp.pt');(out/'qat.last.tmp.pt').replace(statefile)
        if improved:torch.save(state,out/'qat.best.pt')
        write_json(out/'qat-history.json',history)
        print(f'qat_epoch={epoch+1} loss={total/n:.6f} score={score:.6f}',flush=True)
    model.load_state_dict(torch.load(out/'qat.best.pt',map_location='cuda',weights_only=False)['state_dict'])
    model=model.cpu().eval();model.apply(disable_observer)
    # Convert/fold BN first, then express reference quantization as exportable
    # QDQ operations. Exporting unfused QAT ConvBn directly changes weight/BN
    # order and can produce materially different logits.
    reference=exportable_reference(model)
    path=out/'qat.int8.qdq.onnx'
    example=torch.zeros(1,3,192,320)
    torch.onnx.export(reference,example,str(path),input_names=['rgb'],output_names=['logits'],opset_version=17,do_constant_folding=True)
    # Make tensor Constants explicit initializers so scale/zero-point metadata
    # uses the same format as ORT-produced QDQ graphs.
    graph=onnx.load(str(path));removed=[]
    for node in graph.graph.node:
        if node.op_type=='Constant' and len(node.attribute)==1 and node.attribute[0].name=='value':
            tensor=copy.deepcopy(node.attribute[0].t);tensor.name=node.output[0]
            graph.graph.initializer.append(tensor);removed.append(node)
    for node in removed:graph.graph.node.remove(node)
    from onnx import numpy_helper
    initializers={i.name:i for i in graph.graph.initializer}
    quant_params={i for node in graph.graph.node if node.op_type in ('QuantizeLinear','DequantizeLinear')
                  for i in node.input[1:3]}
    for name in quant_params:
        if name in initializers:
            value=numpy_helper.to_array(initializers[name])
            if value.size==1:
                initializers[name].CopyFrom(numpy_helper.from_array(value.reshape(()),name=name))
    # Store constant weight QDQ inputs as actual int8 arrays, not only float
    # weights plus a runtime QuantizeLinear. Activation quantization stays live.
    initializers={i.name:numpy_helper.to_array(i) for i in graph.graph.initializer}
    folded=[]
    for node in graph.graph.node:
        if node.op_type!='QuantizeLinear' or not all(i in initializers for i in node.input):continue
        weight,scale,zp=[initializers[i] for i in node.input]
        if scale.ndim:
            axis=next((a.i for a in node.attribute if a.name=='axis'),1)
            shape=[1]*weight.ndim;shape[axis]=len(scale)
            scale,zp=scale.reshape(shape),zp.reshape(shape)
        limits=np.iinfo(zp.dtype)
        quantized=np.clip(np.rint(weight/scale)+zp,limits.min,limits.max).astype(zp.dtype)
        graph.graph.initializer.append(numpy_helper.from_array(quantized,node.output[0]));folded.append(node)
    for node in folded:graph.graph.node.remove(node)
    onnx.checker.check_model(graph);onnx.save(graph,str(path))
    inv=inventory(path)
    if set(inv['operators'])-{'Conv','Relu','Add','Resize','Constant','Identity','Shape','Slice','Concat',
                             'QuantizeLinear','DequantizeLinear','BatchNormalization'}:
        raise RuntimeError(f'QAT graph adds unreviewed operators: {inv["operators"]}')
    runtime=session(path);parity=0.;probability_error=[];agreements=[]
    with torch.no_grad():
        for ds in validation.values():
            for i in np.linspace(0,len(ds)-1,min(8,len(ds)),dtype=int):
                x,_,_=ds[int(i)];a=reference(x[None]).numpy();b=runtime.run(None,{'rgb':x[None].numpy()})[0]
                parity=max(parity,float(np.max(np.abs(a-b))))
                pa,pb=torch.from_numpy(a).sigmoid(),torch.from_numpy(b).sigmoid()
                probability_error.append(float(torch.mean(torch.abs(pa-pb))))
                agreements.append(float(((pa>=.5)==(pb>=.5)).float().mean()))
    # Quantizer rounding thresholds can amplify tiny float Conv differences.
    # Record actual drift; final choice always uses ONNX domain metrics.
    if np.mean(probability_error)>.01 or np.mean(agreements)<.98:
        raise RuntimeError(f'QAT ONNX/reference drift: max_logit={parity}, probability_MAE={np.mean(probability_error)}, mask_agreement={np.mean(agreements)}')
    torch.jit.trace(reference,example).save(str(out/'qat.reference.torchscript.pt'))
    domains=evaluate_onnx(path,validation)
    result=dict(name='qat',domains=domains,score=domain_score(domains),sha256=file_hash(path),
        activation='QUInt8',method='3 epochs real mixed QAT + FP32 logit regularizer; frozen BN/ranges',
        activation_symmetric=False,history=history,reference_onnx_max_logit_error=parity,
        reference_onnx_probability_mae=float(np.mean(probability_error)),reference_onnx_mask_agreement=float(np.mean(agreements)),
        note='QAT FX checkpoint uses transformed names; restore via prepare_qat_fx or use exported reference TorchScript. Original FP32 best.pt remains intact.')
    write_json(report_file,result)
    return result
