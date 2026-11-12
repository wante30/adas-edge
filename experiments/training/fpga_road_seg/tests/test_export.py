import numpy as np
import torch
import onnx
import pytest
from experiments.training.fpga_road_seg.export import export, multiplier_shift, session, quantization_metadata
from experiments.training.fpga_road_seg.models import build_model
from experiments.training.fpga_road_seg.smoke_data import generate
from experiments.training.fpga_road_seg.dataset_adapter import read_json


def test_fp32_int8_execute_with_train_only_calibration_and_bias_contract(tmp_path):
    manifest=generate(tmp_path/'fixture',count=12)
    torch.manual_seed(15)
    model=build_model('candidate')
    cfg=dict(model='candidate',width=320,height=192,threshold=.5)
    ck=tmp_path/'smoke.pt'
    torch.save(dict(config=cfg,state_dict=model.state_dict(),evidence='synthetic_smoke'),ck)
    with pytest.raises(ValueError,match='synthetic'):
        export(ck,manifest,tmp_path/'forbidden')
    out=tmp_path/'export'
    report=export(ck,manifest,out,calibration_count=3,synthetic_smoke=True)
    assert report['comparison']['torch_onnx_max_logit_error']<1e-3
    assert report['comparison']['samples']==2 and report['calibration_samples']==3
    assert not report['operators_outside_candidate_profile']
    calibration=read_json(out/'calibration-manifest.json')
    assert all(r['split']=='train' for r in calibration)
    quant=read_json(out/'quantization.json')
    assert len(quant['layers'])==14
    for layer in quant['layers']:
        assert layer['weight']['storage_dtype']=='int8' and layer['weight']['axis']==0
        assert layer['activation']['storage_dtype']=='int8'
        assert layer['bias']['storage_dtype']=='int32'
        expected=np.asarray(layer['weight']['scale'])*np.asarray(layer['activation']['scale'])
        np.testing.assert_allclose(layer['bias']['scale'],expected,rtol=1e-6)
        assert layer['requantization']['max_absolute_error']<1e-6
    for name in ('model.fp32.onnx','model.int8.qdq.onnx'):
        onnx.checker.check_model(onnx.load(out/name))
        logits=session(out/name).run(None,{'rgb':np.zeros((1,3,192,320),np.float32)})[0]
        assert logits.shape==(1,4,192,320) and np.isfinite(logits).all()
    # Real QAT export aliases shared zero-points. Metadata must resolve the
    # alias without changing the measured graph or assuming direct initializers.
    graph=onnx.load(out/'model.int8.qdq.onnx')
    node=next(n for n in graph.graph.node if n.op_type=='DequantizeLinear')
    original=node.input[2];alias='shared_zero_point_alias';node.input[2]=alias
    graph.graph.node.insert(0,onnx.helper.make_node('Identity',[original],[alias]))
    aliased=out/'aliased.int8.qdq.onnx';onnx.save(graph,aliased)
    assert quantization_metadata(aliased)==quantization_metadata(out/'model.int8.qdq.onnx')


def test_integer_multiplier_shift_reconstructs_fraction_and_gain():
    values=np.array([.000001,.01,.5,1.,2.,32.])
    result=multiplier_shift(values)
    reconstructed=np.array(result['multiplier'])*np.exp2(-np.array(result['right_shift'],dtype=float))
    np.testing.assert_allclose(reconstructed,values,rtol=1e-8)
