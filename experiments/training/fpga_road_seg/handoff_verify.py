"""Standalone handoff checksum, weights, ONNX, integer and golden self-check."""
import argparse
import csv
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np

try:
    from .handoff_integer import load_array, load_json, sha256, run_integer, compare_arrays
    from .handoff_ort import capture_session
except ImportError:
    from handoff_integer import load_array, load_json, sha256, run_integer, compare_arrays
    from handoff_ort import capture_session


def verify(root, full=True, checksums=True):
    root=Path(root).resolve();start=time.perf_counter();files_checked=0
    if checksums:
        for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            digest,relative=line.split('  ',1);path=root/relative
            if not path.is_relative_to(root) or sha256(path)!=digest:
                raise ValueError(f'checksum mismatch: {relative}')
            files_checked+=1
    identity=load_json(root/'MODEL_IDENTITY.json');contract=load_json(root/'04_量化参数/integer_contract.json')
    model=root/'02_模型与权重/model.int8.qdq.onnx'
    if sha256(model)!=identity['sha256']['int8'] or sha256(root/'02_模型与权重/qat.best.pt')!=identity['sha256']['qat_checkpoint']:
        raise ValueError('final QAT model/checkpoint identity mismatch')
    import onnx
    from onnx import numpy_helper
    m=onnx.load(model);onnx.checker.check_model(m)
    init={a.name:numpy_helper.to_array(a) for a in m.graph.initializer}
    prod={o:n for n in m.graph.node for o in n.output}
    parsed_arrays=0
    for s in contract['stages']:
        if s['op']!='Conv':continue
        weights=load_array(root,s['weights']);bias=load_array(root,s['bias_int32'])
        node=next(n for n in m.graph.node if n.name==s['name'])
        wq=prod[node.input[1]]
        np.testing.assert_array_equal(weights,init[wq.input[0]])
        np.testing.assert_array_equal(load_array(root,s['weight_scale']),init[wq.input[1]])
        np.testing.assert_array_equal(load_array(root,s['weight_zero_point']),init[wq.input[2]])
        np.testing.assert_array_equal(load_array(root,s['original_bias_float32']),init[node.input[2]])
        b=np.rint(init[node.input[2]].astype(np.float64)/(s['input_scale']*init[wq.input[1]].astype(np.float64))).astype('<i4')
        np.testing.assert_array_equal(bias,b)
        for suffix,key in [('multiplier','multipliers'),('right_shift','right_shifts')]:
            parameter=root/'04_量化参数'/f"{s['id']}.{suffix}.bin"
            np.testing.assert_array_equal(np.fromfile(parameter,dtype='<i4'),s[key])
        parsed_arrays+=7
    blobs=load_json(root/'02_模型与权重/weights.offsets.json')
    blob=(root/'02_模型与权重/weights.oihw.int8.bin').read_bytes()
    if len(blob)!=blobs['bytes'] or sha256(root/'02_模型与权重/weights.oihw.int8.bin')!=blobs['sha256']:
        raise ValueError('weight blob identity mismatch')
    for seg in blobs['segments']:
        s=next(v for v in contract['stages'] if v['id']==seg['stage'])
        if blob[seg['offset_bytes']:seg['offset_bytes']+seg['length_bytes']]!=(root/s['weights']['file']).read_bytes():
            raise ValueError('weight offset mismatch')
    for name in ('integer_stages.csv','onnx_nodes.csv'):
        with (root/'03_网络结构与算子'/name).open(encoding='utf-8-sig',newline='') as f:
            if len(list(csv.DictReader(f)))!={'integer_stages.csv':20,'onnx_nodes.csv':93}[name]:
                raise ValueError('structure CSV row count differs')
    capture,_=capture_session(root,contract)
    samples=load_json(root/'05_硬件验证向量/samples.json');results=[]
    for row in samples:
        folder=root/'05_硬件验证向量'/row['id'];sample=load_json(folder/'sample.json')
        arrays=sample['arrays']
        for spec in arrays.values():
            raw=load_array(root,spec)
            np.testing.assert_array_equal(raw,np.load(root/spec['npy_file'],allow_pickle=False));parsed_arrays+=1
        x=load_array(root,arrays['input_fp32']);qin=load_array(root,arrays['input_quantized_u8'])
        rgb=load_array(root,arrays['input_rgb_u8'])
        np.testing.assert_array_equal(x,rgb.astype(np.float32)/np.float32(255))
        lut=load_array(root,contract['input_lut'])
        np.testing.assert_array_equal(qin,lut[rgb])
        logits,stages=capture(x)
        np.testing.assert_array_equal(logits,load_array(root,arrays['output_qdq_logits_fp32']))
        for sid,value in stages.items():np.testing.assert_array_equal(value,load_array(root,arrays['qdq/'+sid]))
        t=time.perf_counter();final,values,acc=run_integer(root,qin)
        np.testing.assert_array_equal(final,load_array(root,arrays['output_integer_uint8']))
        np.testing.assert_array_equal(load_array(root,arrays['qdq_mask_320x192']), (logits[0]>=0).astype(np.uint8))
        np.testing.assert_array_equal(load_array(root,arrays['integer_mask_320x192']),
                                      (final[0]>=contract['stages'][-1]['zero_point']).astype(np.uint8))
        for sid,value in values.items():np.testing.assert_array_equal(value,load_array(root,arrays['integer/'+sid]))
        for sid,value in acc.items():np.testing.assert_array_equal(value,load_array(root,arrays['accumulator/'+sid]))
        results.append(dict(sample=row['id'],integer_seconds=time.perf_counter()-t,integer_stage_outputs=20,
                            integer_accumulators=14,ort_stages=20,status='bit-exact to stored respective reference'))
    checkpoint_status='not requested'
    if full:
        import torch
        spec=importlib.util.spec_from_file_location('handoff_restore',root/'06_参考推理与转换工具/restore_checkpoint.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        qat=module.restore_qat(root/'02_模型与权重/qat.best.pt')
        fpck=torch.load(root/'02_模型与权重/fp32.best.pt',map_location='cpu',weights_only=False)
        fp=module.RoadCandidatePlus();fp.load_state_dict(fpck['state_dict'],strict=True);fp.eval()
        # PyTorch Windows C++ fopen may mishandle Unicode filenames; Python IO
        # preserves the requested Chinese package directory without renaming it.
        traced=torch.jit.load(io.BytesIO((root/'02_模型与权重/qat.reference.torchscript.pt').read_bytes()),map_location='cpu')
        torch.set_num_threads(2)
        with torch.no_grad():
            for name,network in [('qat_checkpoint',qat),('fp32_checkpoint',fp),('qat_torchscript',traced)]:
                output=network(torch.from_numpy(x))
                if output.shape!=(1,4,192,320) or not torch.isfinite(output).all():
                    raise ValueError(f'{name} cannot run')
        checkpoint_status='strict QAT FX/FP32 state_dict loaded; both and exported QAT TorchScript execute finite [1,4,192,320]'
    # Comparator must reject a deliberately corrupted RTL result.
    faulty=final.copy();faulty.flat[0]^=1
    if compare_arrays(final,faulty)['passed']:
        raise ValueError('RTL comparator missed deliberately changed byte')
    return dict(status='PASS',checksum_files=files_checked,parsed_arrays=parsed_arrays,
                model_sha256=sha256(model),checkpoint_status=checkpoint_status,samples=results,
                comparator_negative_control='one altered output byte correctly rejected',seconds=time.perf_counter()-start,
                limitation='software self-check only; no RTL simulation, FPGA resource, timing or hardware accuracy measurements')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--quick',action='store_true',help='skip PyTorch checkpoint restoration only')
    p.add_argument('--report',help='optional new JSON path outside checksum-covered package')
    a=p.parse_args();result=verify(a.root,full=not a.quick)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.report:
        with Path(a.report).open('x',encoding='utf-8') as f:f.write(text)
    print(text)


if __name__=='__main__':main()
