"""Export AVM-SLAM calibration with the supplied, working legacy 4CH ABI.

The archive is the source of the output shape and camera-slot contract.
Geometry is regenerated for the current rig; no output resize, manual flip,
or reuse of the old rig's calibration is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import cv2
import numpy as np

from avm_slam_geometry import ground_maps, blend
from avm_slam_lut import FIELDS, from_maps, encode, decode, Replay, pack565, expand565
from avm_fpga_fixed_model import render, read_lut


GOLDEN = (0, 24, 49, 74, 99)
ROLES = ('front', 'left', 'right', 'rear')
# Calibration camera order is left/front/right/back, not the old FPGA order.
CALIBRATION_TO_FPGA = np.array([1, 0, 2, 3], dtype=np.uint32)
VIDEO_NAMES = ('b106ba02a43a7ce790f378bf10533d36_raw.mp4',
               '47b43f60e2085fac67d6346fb54de39f_raw.mp4',
               '863e9d946a1dc77f6dd1169da2ae28e1_raw.mp4',
               'f42b4a943a47a4a6fb1d6b8a3914cc8d_raw.mp4')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def legacy_parameters(archive):
    params = json.loads(archive.read('4ch/parameters.json'))
    fixed = dict(input_width=640, input_height=320, output_width=480, output_height=640,
                 channel_count=4, coordinate_fraction_bits=4, alpha_fraction_bits=6,
                 alpha_unity=64, lut_entry_bytes=12, lut_entries=480*640,
                 pixel_format='RGB565', binary_endianness='little', pixel_order='row-major')
    for field, expected in fixed.items():
        if params.get(field) != expected:
            raise ValueError(f'Legacy contract mismatch: {field}={params.get(field)!r}, expected {expected!r}')
    if params['slot_camera_names'] != ['front_right', 'side_left', 'side_right', 'rear_center']:
        raise ValueError('This adapter requires legacy slots front/left/right/rear')
    return params


def with_legacy_slots_and_sentinels(fields):
    """Simultaneous ID remap, then reproduce old inactive-field semantics."""
    result = {key: np.array(value, copy=True) for key, value in fields.items()}
    for side in ('a', 'b'):
        active = fields['valid'] != 0
        if side == 'b':
            active = active & (fields['blend'] != 0)
        result[f'cam_{side}'][active] = CALIBRATION_TO_FPGA[fields[f'cam_{side}'][active]]
        result[f'cam_{side}'][~active] = 7
        for name in ('addr', 'fx', 'fy'):
            result[f'{name}_{side}'][~active] = 0
    result['alpha_q'][fields['valid'] == 0] = 0
    return decode(encode(result))


def source_inspection_fields():
    """Four unrotated camera thumbnails, top to bottom slots 0/1/2/3.

480x640 output; each 160-row band displays a 320x160 thumbnail centered in
the 480-wide canvas. Actual 640x320 source samples use exact Q0.4 centres.
"""
    y, x = np.mgrid[:640, :480]
    valid = (x >= 80) & (x < 400)
    fields = {key: np.zeros(480 * 640, np.uint32) for key in FIELDS}
    fields['cam_a'][:] = fields['cam_b'][:] = 7
    fields['valid'] = valid.ravel().astype(np.uint32)
    fields['cam_a'][valid.ravel()] = (y // 160)[valid]
    # u=2*(x-80)+0.5, v=2*(y%160)+0.5; no last-neighbour overflow.
    fields['addr_a'][valid.ravel()] = (2*(y%160)*640 + 2*(x-80))[valid]
    fields['fx_a'][valid.ravel()] = fields['fy_a'][valid.ravel()] = 8
    fields['alpha_q'][valid.ravel()] = 64
    return decode(encode(fields))


def write_lut(folder, fields):
    folder.mkdir(parents=True, exist_ok=True)
    raw = encode(fields)
    raw.tofile(folder/'lut_96bit_le.bin')
    (folder/'lut_readmemh96.hex').write_text(
        ''.join(row[::-1].tobytes().hex().upper()+'\n' for row in raw), encoding='ascii')
    fields['valid'].astype(np.uint8).tofile(folder/'valid_u8.bin')
    fields['blend'].astype(np.uint8).tofile(folder/'blend_u8.bin')
    fields['alpha_q'].astype('<u2').tofile(folder/'alpha_u16_le.bin')
    return decode(np.fromfile(folder/'lut_96bit_le.bin', np.uint8))


def write_words(stem, words):
    words.astype('<u2').tofile(stem.with_suffix('.bin'))
    stem.with_suffix('.hex').write_text(''.join(f'{int(w):04X}\n' for w in words.ravel()), encoding='ascii')


def audit_legacy(archive, out):
    raw = np.frombuffer(archive.read('4ch/lut/lut_96bit_le.bin'), np.uint8)
    fields = decode(raw)
    lines = archive.read('4ch/lut/lut_readmemh96.hex').decode('ascii').splitlines()
    assert b''.join(int(line,16).to_bytes(12,'little') for line in lines) == raw.tobytes()
    results = []
    for frame in GOLDEN:
        prefix = f'4ch/golden/frame_{frame:03d}'
        paths = [next(p for p in archive.namelist() if p.startswith(f'{prefix}/inputs/slot_{slot}_') and p.endswith('.bin')) for slot in range(4)]
        images = [np.frombuffer(archive.read(path), '<u2').reshape(320,640) for path in paths]
        expected = np.frombuffer(archive.read(f'{prefix}/expected_rgb565.bin'), '<u2').reshape(640,480)
        mismatch = int(np.count_nonzero(Replay(fields,(480,640)).render(images) != expected))
        independent = int(np.count_nonzero(render(fields,images,480,640) != expected))
        assert mismatch == independent == 0
        results.append(dict(frame=frame, fast_mismatches=mismatch, independent_mismatches=independent))
    # Minimal unmodified old case for reference RTL regression.
    for name in archive.namelist():
        if name == '4ch/lut/lut_readmemh96.hex' or (name.startswith('4ch/golden/frame_000/') and name.endswith('.hex')):
            dst = out/'reference/legacy_case'/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(archive.read(name))
    calibration = json.loads(archive.read('reference/calibration/cams_lidars.json'))
    result = dict(pc_replay=results, original_lut_bytes=raw.size,
                  original_lut_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                  selected_camera_lenses={name:calibration['cameras'][name]['Lens'] for name in ('front_right','side_left','side_right','rear_center')},
                  inactive_cam_a=7, inactive_cam_b=7,
                  legacy_readme_error='LUT output index is y*480+x; source pixel index is y*640+x')
    save_json(out/'reference/legacy_audit.json',result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--legacy-zip', required=True, type=Path)
    ap.add_argument('--calibration', required=True, type=Path)
    ap.add_argument('--videos', required=True, type=Path)
    ap.add_argument('--reference-delivery', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    with zipfile.ZipFile(args.legacy_zip) as archive:
        original_params = legacy_parameters(archive)
        out.mkdir(parents=True)
        for folder in ('4ch/golden', 'reference', 'preview', 'logs', 'source', 'tools', 'tests', 'rtl', 'diagnostic'):
            (out/folder).mkdir(parents=True, exist_ok=True)
        save_json(out/'reference/old_parameters.json',original_params)
        (out/'reference/old_README_original.md').write_bytes(archive.read('README.md'))
        (out/'reference/verify_fpga_handoff_original.py').write_bytes(archive.read('verify_fpga_handoff.py'))
        old_audit = audit_legacy(archive, out)
    # Reproject onto exactly the old rectangular output and metric extent.
    w,h = original_params['output_width'],original_params['output_height']
    xmin,xmax,ymin,ymax = original_params['bev_bounds_m']
    if xmin != -xmax or ymin != -ymax:
        raise ValueError('Expected symmetric old BEV bounds')
    cv2.setNumThreads(4)
    maps,masks,scores,body,geometry = ground_maps(args.calibration,size=(w,h),extent=(xmax,ymax))
    _,ca,cb,alpha,valid,mix = blend(np.zeros((4,h,w,3),np.uint8),scores)
    calibrated = decode(encode(from_maps(maps,ca,cb,alpha,valid,mix)))
    fixed = write_lut(out/'4ch/lut',with_legacy_slots_and_sentinels(calibrated))
    from_file = read_lut(out/'4ch/lut/lut_96bit_le.bin',w,h)
    assert all(np.array_equal(from_file[k],fixed[k]) for k in fixed)
    # Check geometry against an independent library on this rectangular grid.
    config=json.loads(args.calibration.read_text(encoding='utf-8-sig'))
    xx,yy=np.meshgrid(xmin+(np.arange(w)+.5)*(xmax-xmin)/w, ymax-(np.arange(h)+.5)*(ymax-ymin)/h)
    world=np.stack([xx,yy,np.zeros_like(xx)],axis=-1)
    projection_errors=[]
    for i,c in enumerate(sorted(config['camParams'],key=lambda v:v['camId'])):
        e=np.asarray(c['Extrinsics'],float).reshape(3,4)
        points=world[masks[i]][::491].reshape(1,-1,3)
        rv,_=cv2.Rodrigues(e[:,:3])
        projected,_=cv2.fisheye.projectPoints(points,rv,e[:,3],np.asarray(c['oriIntrinsics'],float).reshape(3,3),np.asarray(c['Distortion'],float))
        uv=(projected.reshape(-1,2)+.5)*np.array([.5,1/3])-.5
        error=float(np.max(abs(uv-maps[i][masks[i]][::491])))
        assert error<1e-4
        projection_errors.append(error)
    diag=write_lut(out/'diagnostic/lut',source_inspection_fields())
    params=dict(original_params)
    params.update(slot_camera_names=['front','left','right','rear'],source_global_ids=[1,0,2,3],
                  calibration_source='user AVM-SLAM stitch.json; old A2D2 calibration not reused',
                  valid_pixels=int(valid.sum()),blend_pixels=int(mix.sum()),
                  source_orientation='raw decoded orientation; no rotation or mirroring',
                  source_files=list(VIDEO_NAMES),input_stride_bytes=1280,output_stride_bytes=960,
                  lut_output_index='y*480+x',source_pixel_index='y*640+x',
                  synchronization_status='four supplied MP4 PTS arrays match; sensor exposure sync unverified',
                  hardware_verified=False)
    save_json(out/'4ch/parameters.json',params)
    shutil.copy2(args.calibration,out/'source/stitch.json')
    shutil.copy2(args.reference_delivery/'source/video_pts.npz',out/'source/video_pts.npz')
    source_audit=json.loads((args.reference_delivery/'source/input_audit.json').read_text(encoding='utf-8'))
    hashes={Path(s['file'].replace('\\','/')).name:s['sha256'] for s in source_audit['streams']}
    for name in VIDEO_NAMES:
        if sha(args.videos/name) != hashes[name]:
            raise ValueError(f'Video changed: {name}')
    pts=np.load(out/'source/video_pts.npz')['input_0']
    ticks=np.rint(pts[:101]*25).astype(int)
    repeats=np.diff(ticks)
    assert len(repeats)==100 and np.all(repeats>0)
    caps=[cv2.VideoCapture(str(args.videos/name)) for name in VIDEO_NAMES]
    if not all(cap.isOpened() for cap in caps):
        raise RuntimeError('Failed to open all four videos')
    video_path=out/'preview/avm_slam_legacy_contract_first_100_frames.mp4'
    ffmpeg=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','25','-i','pipe:0','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video_path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    replay=Replay(fixed,(w,h)); diagnostic=Replay(diag,(w,h))
    results=[];manifest_rows=[]
    try:
        for frame in range(100):
            working=[]
            for cap in caps:
                ok,image=cap.read()
                if not ok:raise RuntimeError(f'Decode failed at frame {frame}')
                working.append(cv2.resize(image,(640,320),interpolation=cv2.INTER_LINEAR))
            words=[pack565(image) for image in working]
            actual=replay.render(words)
            if frame in GOLDEN:
                expected=render(from_file,words,w,h)
                assert np.array_equal(actual,expected)
                # Camera permutation changes ABI only, not projected pixels.
                calibration_order=[words[1],words[0],words[2],words[3]]
                assert np.array_equal(actual,Replay(calibrated,(w,h)).render(calibration_order))
                folder=out/'4ch/golden'/f'frame_{frame:03d}'
                (folder/'inputs').mkdir(parents=True)
                for slot,role in enumerate(ROLES):
                    write_words(folder/'inputs'/f'slot_{slot}_{role}',words[slot])
                    manifest_rows.append([frame,slot,role,frame,str(args.videos/VIDEO_NAMES[slot]),'matching supplied video PTS; exposure sync unverified'])
                write_words(folder/'expected_rgb565',expected)
                cv2.imwrite(str(folder/'expected_rgb565.png'),expand565(expected))
                cv2.imwrite(str(out/'preview'/f'frame_{frame:03d}.png'),expand565(expected))
                results.append(dict(frame=frame,pixels=w*h,independent_mismatches=0,permutation_mismatches=0))
                print(json.dumps(results[-1]),flush=True)
                if frame==0:
                    d=diagnostic.render(words)
                    assert np.array_equal(d,render(diag,words,w,h))
                    write_words(out/'diagnostic/expected_rgb565',d)
                    cv2.imwrite(str(out/'diagnostic/expected_rgb565.png'),expand565(d))
            payload=expand565(actual).tobytes()
            for _ in range(int(repeats[frame])):
                ffmpeg.stdin.write(payload)
    finally:
        for cap in caps:cap.release()
        ffmpeg.stdin.close()
    error=ffmpeg.stderr.read().decode('utf-8',errors='replace')
    if ffmpeg.wait()!=0:raise RuntimeError(error)
    (out/'logs/ffmpeg.log').write_text(error,encoding='utf-8')
    with (out/'4ch/golden/source_manifest.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['frame_index','slot','camera_name','source_index','source_uri','synchronization_evidence']);writer.writerows(manifest_rows)
    # Independent time-spaced checks across the full original sequence.
    long_checks=[]
    for frame in (600,1490,2400,2981):
        folder=args.reference_delivery/'golden'/f'frame_{frame:04d}'
        original_words=[np.fromfile(folder/f'slot_{i}_{role}.bin','<u2').reshape(320,640) for i,role in enumerate(('left','front','right','back'))]
        words=[original_words[1],original_words[0],original_words[2],original_words[3]]
        expected=render(from_file,words,w,h)
        assert np.array_equal(replay.render(words),expected)
        long_checks.append(dict(frame=frame,pixels=w*h,independent_mismatches=0))
        cv2.imwrite(str(out/'preview'/f'long_frame_{frame:04d}.png'),expand565(expected))
    repo=Path(__file__).resolve().parents[1]
    for name in ('export_avm_legacy_contract.py','verify_avm_legacy_contract.py','run_avm_legacy_rtl.py','avm_slam_geometry.py','avm_slam_lut.py','avm_fpga_fixed_model.py'):
        shutil.copy2(repo/'tools'/name,out/'tools'/name)
    shutil.copy2(repo/'tests/test_avm_legacy_contract.py',out/'tests/test_avm_legacy_contract.py')
    shutil.copy2(repo/'rtl/avm_640/avm_4ch_640_stream.v',out/'rtl/avm_4ch_640_stream.v')
    shutil.copy2(repo/'rtl/avm_compat/tb_avm_legacy_contract.v',out/'rtl/tb_avm_legacy_contract.v')
    comparison=dict(old_zip_sha256=sha(args.legacy_zip),legacy_audit=old_audit,
                    compatible_fields={key:params[key] for key in ('input_width','input_height','output_width','output_height','channel_count','coordinate_fraction_bits','alpha_fraction_bits','alpha_unity','lut_entry_bytes','lut_entries','pixel_format','binary_endianness','pixel_order')},
                    slot_roles=list(ROLES),invalid_camera_sentinel=7,
                    previous_delivery=dict(output_size=[640,640],camera_roles=['left','front','right','back'],inactive_camera_sentinel=0),
                    current_calibration_centers=geometry,projection_vs_opencv_max_error=projection_errors,
                    normal_raw_orientation=True,front_rear_rotation_applied=False,
                    old_readme_output_index_corrected='y*480+x (source remains y*640+x)',
                    known_board_status='user reports old LUT works; new physical board result unverified')
    save_json(out/'compatibility_audit.json',comparison)
    save_json(out/'validation.json',dict(status='PC_PASS_REFERENCE_RTL_PENDING',golden=results,long_checks=long_checks,
                                        legacy_regression=old_audit['pc_replay'],bin_bytes=(out/'4ch/lut/lut_96bit_le.bin').stat().st_size,
                                        source_video_frames=100,preview_encoded_frames=int(repeats.sum()),preview_duration_seconds=float(pts[100]-pts[0]),
                                        rtl=[],board_verified=False))
    print(str(out),flush=True)


if __name__=='__main__':
    main()
