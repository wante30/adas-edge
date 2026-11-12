"""Capture the measured ORT CPU optimized graph without disabling its fusion.

Adding outputs to the original QDQ graph blocks fusions and can change results.
We first serialize the optimized graph, then expose its UINT8 tensor boundaries.
The serialized graph is CPU/runtime-specific evidence, never an FPGA artifact.
"""
from pathlib import Path
import copy
import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort


def capture_session(root, contract, optimized_path=None):
    root=Path(root)
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.log_severity_level=3
    if optimized_path is not None:
        options.optimized_model_filepath=str(optimized_path)
        production=ort.InferenceSession(str(root/'02_模型与权重/model.int8.qdq.onnx'), options, providers=['CPUExecutionProvider'])
    else:
        production=ort.InferenceSession(str(root/'02_模型与权重/model.int8.qdq.onnx'), options, providers=['CPUExecutionProvider'])
        optimized_path=root/'02_模型与权重/ort_cpu.optimized.onnx'
    model=onnx.load(str(optimized_path));mappings=[]
    for s in contract['stages']:
        if s['op']=='Input':
            node=next(n for n in model.graph.node if n.op_type=='QuantizeLinear')
        else:
            wanted={'Conv':'QLinearConv','Add':'QLinearAdd','Resize':'Resize'}[s['op']]
            node=next(n for n in model.graph.node if n.op_type==wanted and n.name.split('_token_')[0]==s['name'])
        mappings.append((s,node.output[0]))
    original_outputs={v.name for v in model.graph.output}
    for s,name in mappings:
        if name not in original_outputs:
            model.graph.output.append(helper.make_tensor_value_info(name,onnx.TensorProto.UINT8,None))
    trace_options=ort.SessionOptions();trace_options.intra_op_num_threads=4;trace_options.inter_op_num_threads=1
    trace_options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    trace_options.log_severity_level=3
    runtime=ort.InferenceSession(model.SerializeToString(),trace_options,providers=['CPUExecutionProvider'])
    def run(x):
        result=runtime.run(None,{'rgb':x})
        outputs=dict(zip([v.name for v in runtime.get_outputs()],result))
        logits=production.run(None,{'rgb':x})[0]
        np.testing.assert_array_equal(outputs['logits'],logits)
        stages={}
        for s,name in mappings:
            y=outputs[name]
            if list(y.shape)!=s['shape']:
                y=y.transpose(0,3,1,2)
            if list(y.shape)!=s['shape']:
                raise ValueError('ORT trace layout does not match extracted NCHW')
            stages[s['id']]=np.ascontiguousarray(y)
        return logits,stages
    return run, model
