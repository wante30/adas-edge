"""Portable v1 integer CNN contract. Runtime needs NumPy only; no ONNX/PyTorch.

This is an explicitly derived RTL reference, NOT a bit-exact implementation of
floating QDQ Conv/bias/Add. Exported QAT weights/ranges are never recalibrated.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rne_shift(value, shift):
    """Signed round-to-nearest, ties-to-even; shift >= 0, exact integer math."""
    value = np.asarray(value, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    if np.any(shift < 0) or np.any(shift > 62):
        raise ValueError('shift outside [0,62]')
    if np.any(value == np.iinfo(np.int64).min):
        raise OverflowError('abs(INT64_MIN) unsupported')
    a = np.abs(value)
    quotient = np.right_shift(a, shift)
    remainder = a - np.left_shift(quotient, shift)
    half = np.left_shift(np.ones_like(shift), np.maximum(shift - 1, 0))
    up = (shift > 0) & ((remainder > half) | ((remainder == half) & ((quotient & 1) != 0)))
    return np.where(value < 0, -(quotient + up), quotient + up)


def requantize(acc, multipliers, shifts, zero_point, relu=False):
    acc = np.asarray(acc, dtype=np.int64)
    if relu:
        acc = np.maximum(acc, 0)
    m = np.asarray(multipliers, np.int64).reshape(1, -1, 1, 1)
    s = np.asarray(shifts, np.int64).reshape(1, -1, 1, 1)
    if np.any(np.abs(acc) > np.iinfo(np.int64).max // np.maximum(m, 1)):
        raise OverflowError('requantization product overflows INT64')
    return np.clip(rne_shift(acc * m, s) + zero_point, 0, 255).astype(np.uint8)


def conv_accumulate(x, weights, bias, input_zero_point, strides, pads, groups):
    """Exact grouped OIHW cross-correlation, INT64 oracle, checked INT32 output."""
    x = np.asarray(x, np.int64) - input_zero_point
    weights = np.asarray(weights, np.int64)
    bias = np.asarray(bias, np.int64)
    n, cin, ih, iw = x.shape
    cout, cgroup, kh, kw = weights.shape
    sy, sx = strides
    pt, pl, pb, pr = pads
    if cin != cgroup * groups or cout % groups:
        raise ValueError('invalid convolution groups')
    x = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=0)
    oh, ow = (ih + pt + pb - kh) // sy + 1, (iw + pl + pr - kw) // sx + 1
    y = np.broadcast_to(bias[None, :, None, None], (n, cout, oh, ow)).copy()
    og = cout // groups
    if groups == cin == cout:
        for ky in range(kh):
            for kx in range(kw):
                y += x[:, :, ky:ky + oh * sy:sy, kx:kx + ow * sx:sx] * weights[:, 0, ky, kx][None, :, None, None]
    else:
        for g in range(groups):
            a, b = g * cgroup, g * og
            for ky in range(kh):
                for kx in range(kw):
                    patch = x[:, a:a + cgroup, ky:ky + oh * sy:sy, kx:kx + ow * sx:sx]
                    y[:, b:b + og] += np.einsum('oi,nihw->nohw', weights[b:b + og, :, ky, kx], patch,
                                               dtype=np.int64, optimize=False)
    if y.min() < -(1 << 31) or y.max() >= (1 << 31):
        raise OverflowError('INT32 accumulator overflow; no wrap/saturation allowed')
    return y.astype('<i4')


def nearest(x, output_shape):
    """ONNX asymmetric/floor coordinates, integer indexing, no interpolation."""
    oh, ow = output_shape[-2:]
    iy = np.arange(oh, dtype=np.int64) * x.shape[-2] // oh
    ix = np.arange(ow, dtype=np.int64) * x.shape[-1] // ow
    return x[:, :, iy][:, :, :, ix].copy()


def integer_add(a, b, stage):
    # Align both branches to the SAME Q31 domain, sum BEFORE one RNE operation.
    za, zb = stage['input_zero_points']
    ma, mb = stage['multipliers']
    value = (a.astype(np.int64) - za) * ma + (b.astype(np.int64) - zb) * mb
    return np.clip(rne_shift(value, stage['right_shift']) + stage['zero_point'], 0, 255).astype(np.uint8)


def load_array(root, spec):
    path = Path(root) / spec['file']
    if sha256(path) != spec['sha256']:
        raise ValueError(f'array checksum mismatch: {path}')
    a = np.fromfile(path, dtype=spec['dtype'])
    if a.size != int(np.prod(spec['shape'])):
        raise ValueError(f'array length mismatch: {path}')
    return a.reshape(spec['shape'])


def run_integer(root, quantized_input, capture=True):
    root = Path(root)
    contract = load_json(root / '04_量化参数/integer_contract.json')
    if contract['version'] != 'candidate-plus-integer-v1':
        raise ValueError('unsupported integer contract')
    x = np.asarray(quantized_input)
    if x.shape != (1, 3, 192, 320) or x.dtype != np.uint8:
        raise ValueError('expected quantized UINT8 NCHW [1,3,192,320]')
    values = {'s00': x.copy()}
    accumulators = {}
    for stage in contract['stages'][1:]:
        inputs = [values[k] for k in stage['inputs']]
        if stage['op'] == 'Conv':
            weights = load_array(root, stage['weights'])
            bias = load_array(root, stage['bias_int32'])
            acc = conv_accumulate(inputs[0], weights, bias, stage['input_zero_point'],
                                  stage['strides'], stage['pads'], stage['groups'])
            y = requantize(acc, stage['multipliers'], stage['right_shifts'], stage['zero_point'], stage['relu'])
            if capture:
                accumulators[stage['id']] = acc
        elif stage['op'] == 'Resize':
            y = nearest(inputs[0], stage['shape'])
        elif stage['op'] == 'Add':
            y = integer_add(inputs[0], inputs[1], stage)
        else:
            raise ValueError(f"unsupported stage {stage['op']}")
        if list(y.shape) != stage['shape']:
            raise ValueError('stage shape mismatch')
        values[stage['id']] = y
    return values[contract['stages'][-1]['id']], values, accumulators


def compare_arrays(expected, actual, tolerance=0):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError(f'shape/dtype differ: {expected.shape}/{expected.dtype} vs {actual.shape}/{actual.dtype}')
    error = np.abs(expected.astype(np.int64) - actual.astype(np.int64))
    indices = np.argwhere(error > tolerance)
    return dict(passed=not len(indices), max_abs_error=int(error.max()), mismatched=int(len(indices)),
                total=int(error.size), first_mismatch=indices[0].tolist() if len(indices) else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    run = sub.add_parser('infer')
    run.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    run.add_argument('--input', required=True, help='quantized UINT8 NCHW .npy')
    run.add_argument('--output', required=True, help='new directory, never overwritten')
    compare = sub.add_parser('compare')
    compare.add_argument('--expected', required=True, help='NPY reference')
    compare.add_argument('--actual', required=True, help='NPY or raw binary with reference shape/dtype')
    compare.add_argument('--tolerance', type=int, default=0)
    layout = sub.add_parser('convert-weight')
    layout.add_argument('--input', required=True, help='OIHW NPY, includes I/group for grouped Conv')
    layout.add_argument('--output', required=True, help='new raw .bin')
    layout.add_argument('--layout', choices=['OIHW', 'OHWI', 'HWIO'], required=True)
    a = p.parse_args()
    if a.command == 'infer':
        out = Path(a.output);out.mkdir(parents=True, exist_ok=False)
        final, values, acc = run_integer(a.root, np.load(a.input, allow_pickle=False))
        np.save(out / 'output_uint8.npy', final);final.tofile(out / 'output_uint8.bin')
        for key, v in {**values, **{k + '_acc': v for k, v in acc.items()}}.items():
            np.save(out / (key + '.npy'), v)
        print(json.dumps(dict(output_shape=list(final.shape), output_sha256=sha256(out / 'output_uint8.bin'))))
    elif a.command == 'compare':
        expected = np.load(a.expected, allow_pickle=False)
        path = Path(a.actual)
        actual = np.load(path, allow_pickle=False) if path.suffix == '.npy' else np.fromfile(path, expected.dtype).reshape(expected.shape)
        report = compare_arrays(expected, actual, a.tolerance)
        print(json.dumps(report));raise SystemExit(0 if report['passed'] else 1)
    else:
        x = np.load(a.input, allow_pickle=False)
        if x.dtype != np.int8 or x.ndim != 4:
            raise ValueError('weights must be INT8 OIHW')
        permutation = {'OIHW': (0, 1, 2, 3), 'OHWI': (0, 2, 3, 1), 'HWIO': (2, 3, 1, 0)}[a.layout]
        y = np.ascontiguousarray(x.transpose(permutation))
        with Path(a.output).open('xb') as f:
            f.write(y.tobytes())
        print(json.dumps(dict(layout=a.layout, shape=list(y.shape), bytes=y.nbytes, sha256=sha256(a.output))))


if __name__ == '__main__':
    main()
