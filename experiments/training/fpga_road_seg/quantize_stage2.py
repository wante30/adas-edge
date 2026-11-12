"""Balanced TRAIN calibration and domain-separated measured PTQ search."""
from pathlib import Path
import hashlib
import time

import numpy as np
import onnx
import torch
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from . import CLASSES
from .dataset_adapter import RoadDataset, file_hash, load_manifest, read_json, write_json
from .engine import Metrics
from .export import ManifestReader, inventory, quantization_metadata, session


def balanced_calibration(rows, per_domain=96):
    selected = []
    for source in ('bdd100k', 'rlmd'):
        group = sorted([r for r in rows if r['split']=='train' and r['source']==source],
                       key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        if source == 'rlmd':
            positive = [r for r in group if r['positive_pixels'][3] > 0]
            negative = [r for r in group if r['positive_pixels'][3] == 0]
            selected.extend(positive[:per_domain//2] + negative[:per_domain//2])
        else:
            selected.extend(group[:per_domain])
    return selected


def domain_score(reports):
    b, r = reports['bdd100k']['classes'], reports['rlmd']['classes']
    return .2 * sum(b[c][m] or 0 for c,m in [('drivable_area','iou'),('solid_lane','f1'),('dashed_lane','f1')]) + .4 * (r['crosswalk']['iou'] or 0)


def maximum_drop(reference, candidate):
    keys=[('bdd100k','drivable_area','iou'),('bdd100k','solid_lane','f1'),('bdd100k','dashed_lane','f1'),
          ('rlmd','crosswalk','iou'),('rlmd','crosswalk','f1'),('rlmd','crosswalk','recall')]
    return max(reference[s]['classes'][c][m]-candidate[s]['classes'][c][m] for s,c,m in keys
               if reference[s]['classes'][c][m] is not None and candidate[s]['classes'][c][m] is not None)


@torch.no_grad()
def evaluate_onnx(path, datasets):
    runtime = session(path); reports = {}
    for source, ds in datasets.items():
        metrics = Metrics(4)
        for x,y,v in ds:
            logits = torch.from_numpy(runtime.run(None, {'rgb': x[None].numpy()})[0])
            metrics.update(logits.sigmoid(), y[None], v[None])
        reports[source] = dict(samples=len(ds), classes=metrics.report(CLASSES))
    return reports


def quantize(model, checkpoint, manifest, output, cfg, force_qat=False):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint_sha256=file_hash(checkpoint), manifest_sha256=file_hash(manifest))
    marker = out/'search.json'
    if marker.exists():
        result = read_json(marker)
        if any(result[k] != v for k,v in identity.items()):
            raise ValueError('PTQ output identity mismatch')
        if force_qat and not result.get('qat_requested'):
            raise ValueError('completed PTQ-only output preserved; use an independent QAT output directory')
        return result
    model = model.cpu().eval()
    fp = out/'model.fp32.onnx'; prepared = out/'model.preprocessed.onnx'
    example = torch.zeros(1,3,cfg['height'],cfg['width'])
    torch.onnx.export(model, example, str(fp), input_names=['rgb'], output_names=['logits'], opset_version=17, do_constant_folding=True)
    onnx.checker.check_model(str(fp))
    quant_pre_process(str(fp), str(prepared), skip_symbolic_shape=True)
    rows = load_manifest(manifest)
    calibration = balanced_calibration(rows)
    write_json(out/'calibration-manifest.json', [dict(id=r['id'], source=r['source'], group_id=r['group_id'],
        split=r['split'], image_sha256=file_hash(r['image'])) for r in calibration])
    validation = {}
    for source in ('bdd100k','rlmd'):
        ds = RoadDataset(manifest, 'val', (cfg['width'],cfg['height']))
        ds.rows = sorted([r for r in ds.rows if r['source']==source], key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        ds.rows = ds.rows[:min(len(ds), 1024 if source=='bdd100k' else 10000)]
        validation[source] = ds
    # Verify real-image parity independently from quantized accuracy.
    fp_session = session(fp); parity = 0.
    for ds in validation.values():
        for i in np.linspace(0,len(ds)-1,min(12,len(ds)),dtype=int):
            x,_,_ = ds[int(i)]
            with torch.no_grad():
                ref = model(x[None]).numpy()
            parity = max(parity, float(np.max(np.abs(ref - fp_session.run(None, {'rgb':x[None].numpy()})[0]))))
    if parity > 1e-3:
        raise RuntimeError(f'ONNX parity failed: {parity}')
    write_json(out/'fp32-operators.json', inventory(fp))
    fp_metrics = evaluate_onnx(fp, validation)
    variants = [
        ('s8_minmax', QuantType.QInt8, CalibrationMethod.MinMax, True, {}),
        ('s8_percentile9999', QuantType.QInt8, CalibrationMethod.Percentile, True, {'CalibPercentile':99.99}),
        ('u8_percentile9999', QuantType.QUInt8, CalibrationMethod.Percentile, False, {'CalibPercentile':99.99}),
        ('u8_percentile999', QuantType.QUInt8, CalibrationMethod.Percentile, False, {'CalibPercentile':99.9}),
        ('s8_entropy', QuantType.QInt8, CalibrationMethod.Entropy, True, {}),
    ]
    results, failures = [], []
    for name, activation, method, symmetric, extra in variants:
        path = out/f'{name}.int8.qdq.onnx'; report_file = out/f'{name}.json'
        try:
            if report_file.exists():
                result = read_json(report_file)
            else:
                quantize_static(str(prepared), str(path), ManifestReader(calibration,(cfg['width'],cfg['height'])),
                    quant_format=QuantFormat.QDQ, activation_type=activation, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=['Conv'], calibrate_method=method,
                    extra_options=dict(ActivationSymmetric=symmetric, WeightSymmetric=True, **extra))
                onnx.checker.check_model(str(path))
                metrics = evaluate_onnx(path, validation)
                result = dict(name=name, domains=metrics, score=domain_score(metrics), sha256=file_hash(path),
                              activation=str(activation), method=str(method), activation_symmetric=symmetric)
                write_json(report_file, result)
            results.append(result)
            print(f'ptq_variant={name} score={result["score"]:.6f}', flush=True)
        except Exception as exc:
            failures.append(dict(name=name, error=repr(exc)))
            write_json(out/'failures.json', failures)
    if not results:
        raise RuntimeError(f'all PTQ variants failed: {failures}')
    def select():
        eligible=[r for r in results if maximum_drop(fp_metrics,r['domains'])<=.02]
        return (max(eligible,key=lambda r:r['score']) if eligible else
                min(results,key=lambda r:(maximum_drop(fp_metrics,r['domains']),-r['score'])))
    selected = select()
    if force_qat or maximum_drop(fp_metrics, selected['domains']) > .02:
        print('Starting bounded real-data QAT; selection remains on internal validation only',flush=True)
        try:
            from .qat_stage2 import run_qat
            results.append(run_qat(checkpoint,manifest,out,calibration,validation))
            selected=select()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failures.append(dict(name='qat',error=repr(exc)))
            write_json(out/'failures.json',failures)
    selected_path = out/f'{selected["name"]}.int8.qdq.onnx'
    import shutil
    shutil.copyfile(selected_path, out/'model.int8.qdq.onnx')
    write_json(out/'quantization.json', quantization_metadata(selected_path))
    write_json(out/'int8-operators.json', inventory(selected_path))
    full = {}
    for source, split in [('bdd100k','test'), ('rlmd','val')]:
        ds = RoadDataset(manifest,split,(cfg['width'],cfg['height']))
        ds.rows = [r for r in ds.rows if r['source']==source]
        full[source] = ds
    full_comparison = dict(fp32=evaluate_onnx(fp,full), int8=evaluate_onnx(selected_path,full))
    write_json(out/'comparison.json',full_comparison)
    report = dict(**identity, torch_onnx_max_logit_error=parity, calibration_samples=len(calibration),
                  calibration='96 BDD + up to 48 RLMD crosswalk positive/48 negative TRAIN samples, no validation/test calibration',
                  selection='Internal BDD 1024 + all independent RLMD validation. Prefer candidates with <=2pp loss per key task, then weighted score; otherwise minimize worst task loss. Held-out BDD never used for selection.',
                  fp32_validation=fp_metrics, variants=results, failures=failures, selected=selected['name'],qat_requested=force_qat,
                  comparison=full_comparison, integer_scope='All Conv W8A8 QDQ. Add/Resize/IO float. Unsigned asymmetric activation may require zero-point correction hardware. Not a full-integer FPGA artifact.')
    write_json(marker, report)
    return report
