"""Compensate 180-degree input orientation for AVM slots 1/front and 3/rear.

Output pixel order, camera IDs, blend weights and slots 0/2 are unchanged.
This adapts the existing 640x320 source / 640x640 output 96-bit contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avm_slam_lut import decode, encode, validate, Replay, expand565
from avm_fpga_fixed_model import render


STEM = 'avm_slam_front_rear_rot180_96bit'
ROLES = ('left', 'front', 'right', 'back')


def rotate_input_coordinates(fields):
    """Map selected samples to (639-u,319-v), retaining Q0.4 exactly.

Integer and fractional coordinates must be transformed together. Merely
subtracting the top-left pixel address from 204799 is incorrect when either
fraction is nonzero. Reject an unrepresentable border rather than clamp it.
"""
    validate(fields)
    result = {key: np.array(value, copy=True) for key, value in fields.items()}
    for side in ('a', 'b'):
        active = fields['valid'] != 0
        if side == 'b':
            active = active & (fields['blend'] != 0)
        selected = active & np.isin(fields[f'cam_{side}'], (1, 3))
        address = fields[f'addr_{side}'][selected].astype(np.int64)
        qx = (639 - address % 640) * 16 - fields[f'fx_{side}'][selected].astype(np.int64)
        qy = (319 - address // 640) * 16 - fields[f'fy_{side}'][selected].astype(np.int64)
        x, fx = np.divmod(qx, 16)
        y, fy = np.divmod(qy, 16)
        if np.any((x < 0) | (x >= 639) | (y < 0) | (y >= 319)):
            raise ValueError('Rotated sample has an unrepresentable bilinear border')
        result[f'addr_{side}'][selected] = y * 640 + x
        result[f'fx_{side}'][selected] = fx
        result[f'fy_{side}'][selected] = fy
    validate(result)
    return result


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_words(stem, words):
    words.astype('<u2').tofile(stem.with_suffix('.bin'))
    stem.with_suffix('.hex').write_text(
        ''.join(f'{int(word):04x}\n' for word in words.ravel()), encoding='ascii')


def png(path, words):
    Image.fromarray(expand565(words)[..., ::-1]).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True, type=Path, help='Original delivery directory')
    parser.add_argument('--output', required=True, type=Path, help='New, nonexistent directory')
    parser.add_argument('--iverilog', type=Path)
    parser.add_argument('--vvp', type=Path)
    parser.add_argument('--rtl-timeout', type=float, default=300, help='Seconds allowed per software RTL simulation')
    parser.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/msyh.ttc'))
    args = parser.parse_args()
    if bool(args.iverilog) != bool(args.vvp):
        parser.error('Supply both --iverilog and --vvp or neither')
    source, out = args.original.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    original_path = source / 'lut/avm_slam_96bit_le.bin'
    raw = np.fromfile(original_path, np.uint8)
    if raw.size != 640 * 640 * 12:
        raise ValueError('Expected 409600 entries of 12 bytes')
    original = decode(raw)
    changed = rotate_input_coordinates(original)
    packed = encode(changed)
    # An orientation transform must be reversible without requantization.
    assert np.array_equal(encode(rotate_input_coordinates(changed)), raw.reshape(-1, 12))
    for folder in ('lut', 'preview', 'golden', 'logs', 'tools', 'tests', 'rtl'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    bin_path = out / 'lut' / f'{STEM}_le.bin'
    hex_path = out / 'lut' / f'{STEM}.hex'
    packed.tofile(bin_path)
    hex_path.write_text(''.join(row[::-1].tobytes().hex() + '\n' for row in packed), encoding='ascii')
    assert b''.join(int(line, 16).to_bytes(12, 'little') for line in hex_path.read_text().splitlines()) == packed.tobytes()
    # Replay the bytes read back from the delivered BIN, not pre-encoded arrays.
    decoded = decode(np.fromfile(bin_path, np.uint8))
    baseline, corrected = Replay(original), Replay(decoded)
    for field in ('cam_a', 'cam_b', 'alpha_q', 'valid', 'blend'):
        assert np.array_equal(decoded[field], original[field])
    modified_samples = {}
    for side in ('a', 'b'):
        selected = (original['valid'] != 0) & np.isin(original[f'cam_{side}'], (1, 3))
        if side == 'b':
            selected &= original['blend'] != 0
        for field in (f'addr_{side}', f'fx_{side}', f'fy_{side}'):
            assert np.array_equal(decoded[field][~selected], original[field][~selected])
        modified_samples[side] = int(np.count_nonzero(selected))
    repo = Path(__file__).resolve().parents[1]
    for name in ('export_avm_front_rear_rot180.py', 'avm_slam_lut.py', 'avm_fpga_fixed_model.py'):
        shutil.copy2(repo / 'tools' / name, out / 'tools' / name)
    shutil.copy2(repo / 'tests/test_avm_front_rear_rot180.py', out / 'tests/test_avm_front_rear_rot180.py')
    for name in ('avm_4ch_640_stream.v', 'tb_avm_4ch_640_stream.v'):
        shutil.copy2(source / 'rtl' / name, out / 'rtl' / name)
    if args.iverilog:
        command = [str(args.iverilog), '-g2012', '-s', 'tb_avm_4ch_640_stream', '-o',
                   'logs/avm_tb.vvp', 'rtl/avm_4ch_640_stream.v', 'rtl/tb_avm_4ch_640_stream.v']
        compile_result = subprocess.run(command, cwd=out, capture_output=True, text=True, check=True)
        (out / 'logs/compile.log').write_text(compile_result.stdout + compile_result.stderr, encoding='utf-8')
    rows = []
    for frame in (0, 600, 1490, 2400, 2981):
        src = source / 'golden' / f'frame_{frame:04d}'
        dst = out / 'golden' / f'frame_{frame:04d}'
        dst.mkdir()
        normal = [np.fromfile(src / f'slot_{i}_{role}.bin', '<u2').reshape(320, 640)
                  for i, role in enumerate(ROLES)]
        rotated = [np.ascontiguousarray(word[::-1, ::-1]) if i in (1, 3) else word.copy()
                   for i, word in enumerate(normal)]
        expected = np.fromfile(src / 'expected_rgb565.bin', '<u2').reshape(640, 640)
        old_result = baseline.render(normal)
        faulty = baseline.render(rotated)
        actual = corrected.render(rotated)
        independent = render(decoded, rotated, 640, 640)
        assert np.array_equal(old_result, expected)
        assert np.array_equal(actual, expected) and np.array_equal(actual, independent)
        for i, role in enumerate(ROLES):
            save_words(dst / f'slot_{i}_{role}', rotated[i])
        save_words(dst / 'expected_rgb565', expected)
        png(out / 'preview' / f'frame_{frame:04d}_pc_original.png', expected)
        png(out / 'preview' / f'frame_{frame:04d}_simulated_rotation_uncorrected.png', faulty)
        png(out / 'preview' / f'frame_{frame:04d}_simulated_rotation_corrected.png', actual)
        row = dict(frame=frame, pixels=409600,
                   mismatches_after_correction=int(np.count_nonzero(actual != expected)),
                   mismatches_before_correction=int(np.count_nonzero(faulty != expected)))
        if args.vvp:
            base = f'golden/frame_{frame:04d}'
            command = [str(args.vvp), 'logs/avm_tb.vvp', f'+LUT=lut/{STEM}.hex']
            command += [f'+SRC{i}={base}/slot_{i}_{role}.hex' for i, role in enumerate(ROLES)]
            command += [f'+EXPECTED={base}/expected_rgb565.hex']
            try:
                run = subprocess.run(command, cwd=out, capture_output=True, text=True, check=True, timeout=args.rtl_timeout)
            except subprocess.TimeoutExpired as error:
                output = error.stdout or b''
                if isinstance(output, bytes):
                    output = output.decode('utf-8', errors='replace')
                (out / 'logs' / f'rtl_frame_{frame:04d}_timeout.log').write_text(output, encoding='utf-8')
                raise
            log = run.stdout + run.stderr
            (out / 'logs' / f'rtl_frame_{frame:04d}.log').write_text(log, encoding='utf-8')
            if 'PASS' not in log or 'FAIL' in log:
                raise AssertionError(log)
            row['rtl_log'] = log.strip()
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    canvas = Image.new('RGB', (1920, 714), '#f4f5f7')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(args.font), 25)
    labels = ('PC 原始四路 + 原 LUT', '模拟前后两路倒置 + 原 LUT', '模拟前后两路倒置 + 修正版 LUT')
    suffixes = ('pc_original', 'simulated_rotation_uncorrected', 'simulated_rotation_corrected')
    for i, (label, suffix) in enumerate(zip(labels, suffixes)):
        draw.text((i * 640 + 16, 16), label, font=font, fill='#222222')
        canvas.paste(Image.open(out / 'preview' / f'frame_0000_{suffix}.png'), (i * 640, 64))
    canvas.save(out / 'preview/pc_orientation_comparison.png')
    manifest = dict(status='PC_AND_REFERENCE_RTL_PASS' if args.vvp else 'PC_PASS_RTL_NOT_RUN',
                    purpose='Compensate 180-degree source orientation only for slots 1 and 3',
                    source_lut_sha256=digest(original_path), output_lut_sha256=digest(bin_path),
                    output_lut_bytes=bin_path.stat().st_size, entries=409600, bits_per_entry=96,
                    output_size=[640, 640], source_size=[640, 320],
                    camera_slots=dict(enumerate(ROLES)), corrected_slots=[1, 3], unchanged_slots=[0, 2],
                    source_coordinate_transform='u_new=639-u_old, v_new=319-v_old, including Q0.4 fractions',
                    output_pixel_order='unchanged, row-major', modified_samples=modified_samples,
                    controls_and_unselected_samples_unchanged=True, involution_exact=True,
                    bin_hex_roundtrip_exact=True, frames=rows,
                    limits=['PC inputs synthetically rotate front/rear by 180 degrees; not a board capture',
                            'Original dataset calibration and RGB565 640x320 input contract still apply',
                            'Does not correct channel swaps, wrong stride, DDR corruption or HDMI stretching',
                            'No board/DDR/PCIe/HDMI validation performed'])
    (out / 'validation.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(str(out), flush=True)


if __name__ == '__main__':
    main()
