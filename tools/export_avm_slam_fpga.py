"""Export the supplied AVM-SLAM sequence using the project's 96-bit AVM ABI.

This is an offline PC implementation. It does not program a board.
Run with --help for explicit source paths. Output is a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np

from avm_slam_geometry import ground_maps, warp_inputs, blend
from avm_slam_lut import from_maps, write_lut, Replay, pack565, expand565, decode
from avm_fpga_fixed_model import read_lut, render

NAMES = ['47b43f60e2085fac67d6346fb54de39f_raw.mp4',
         'b106ba02a43a7ce790f378bf10533d36_raw.mp4',
         '863e9d946a1dc77f6dd1169da2ae28e1_raw.mp4',
         'f42b4a943a47a4a6fb1d6b8a3914cc8d_raw.mp4']
ROLES = ['left','front','right','back']
GOLDEN = [0,600,1490,2400,2981]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def write_words(path,words):
    words.astype('<u2').tofile(path.with_suffix('.bin'))
    with path.with_suffix('.hex').open('w',encoding='ascii',newline='\n') as f:
        f.write('\n'.join(f'{int(w):04x}' for w in words.ravel())+'\n')


def float_reference(images,maps,ca,cb,alpha,valid):
    """True float bilinear reference, without OpenCV interpolation table."""
    flat=np.stack(images).reshape(-1,3).astype(np.float32)
    sampled=[]
    for cam in [ca,cb]:
        uv=np.take_along_axis(maps,cam[None,...,None],axis=0)[0]
        x=np.floor(uv[...,0]).astype(int);y=np.floor(uv[...,1]).astype(int)
        fx=uv[...,0]-x;fy=uv[...,1]-y
        addr=cam.astype(int)*640*320+y*640+x
        side=np.zeros((*x.shape,3),np.float64)
        for offset,w in [(0,(1-fx)*(1-fy)),(1,fx*(1-fy)),(640,(1-fx)*fy),(641,fx*fy)]:
            side+=flat[addr+offset]*w[...,None]
        sampled.append(side)
    result=np.clip(np.floor(sampled[0]*alpha[...,None]+sampled[1]*(1-alpha[...,None])+.5),0,255).astype(np.uint8)
    result[~valid]=0
    return result


def metric(actual,reference,valid):
    diff=actual.astype(float)[valid]-reference.astype(float)[valid]
    mse=float(np.mean(diff**2))
    return dict(mae=float(np.mean(abs(diff))),psnr_db=float(10*np.log10(255**2/mse)) if mse else None,
                max_abs=int(np.max(abs(diff))))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--param',type=Path,required=True)
    ap.add_argument('--videos',type=Path,required=True)
    ap.add_argument('--audit-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--no-video',action='store_true')
    args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    for name in ['preview','golden','source','logs']: (out/name).mkdir()
    cv2.setNumThreads(4)
    src=[args.videos/n for n in NAMES]
    audit=json.loads((args.audit_root/'input_audit.json').read_text(encoding='utf-8'))
    hashes={str(s['file']).replace('\\','/').rsplit('/',1)[-1]:s['sha256'] for s in audit['streams']}
    for p in src:
        if sha(p)!=hashes[p.name]:raise ValueError(f'Source changed: {p}')
    pts_archive=np.load(args.audit_root/'video_pts.npz')
    pts=pts_archive['input_0']
    if not all(np.array_equal(pts,pts_archive[f'input_{i}']) for i in range(4)):
        raise ValueError('Video timestamps differ')
    ticks=np.rint(pts*25).astype(int)
    if len(pts)!=2982 or not np.allclose(ticks/25,pts):raise ValueError('Unexpected source timing')
    repeats=np.diff(np.r_[ticks,7466])
    if np.any(repeats<=0):raise ValueError('Nonmonotonic source timing')
    shutil.copy2(args.calibration,out/'source/stitch.json')
    shutil.copy2(args.param,out/'source/param.yaml')
    shutil.copy2(args.audit_root/'input_audit.json',out/'source/input_audit_before_calibration.json')
    shutil.copy2(args.audit_root/'video_pts.npz',out/'source/video_pts.npz')
    maps,masks,scores,body,geometry=ground_maps(args.calibration)
    dummy=np.zeros((4,640,640,3),np.uint8)
    _,ca,cb,alpha,valid,mix=blend(dummy,scores)
    fields=from_maps(maps,ca,cb,alpha,valid,mix)
    decoded=write_lut(out/'lut',fields)
    from_file=read_lut(out/'lut/avm_slam_96bit_le.bin',640,640)
    assert all(np.array_equal(decoded[k],from_file[k]) for k in decoded)
    # Check the actual hexadecimal stream, not just the in-memory packer.
    hex_lines=(out/'lut/avm_slam_96bit.hex').read_text().splitlines()
    bin_data=(out/'lut/avm_slam_96bit_le.bin').read_bytes()
    assert b''.join(int(line,16).to_bytes(12,'little') for line in hex_lines)==bin_data
    del hex_lines,bin_data
    decode(np.fromfile(out/'lut/avm_slam_96bit_le.bin',np.uint8).reshape(-1,12))
    replay=Replay(from_file)
    np.savez_compressed(out/'source/float_inverse_maps.npz',maps=maps,scores=scores,body=body)
    palette=np.array([[230,150,40],[70,210,80],[40,130,235],[200,80,200]],np.uint8)
    ownership=palette[ca];ownership[~valid]=0
    cv2.imwrite(str(out/'preview/camera_ownership.png'),ownership)
    cv2.imwrite(str(out/'preview/coverage.png'),valid.astype(np.uint8)*255)
    cv2.imwrite(str(out/'preview/overlap.png'),mix.astype(np.uint8)*255)
    # Cross-check our projection formula with the independent OpenCV API.
    yy,xx=np.mgrid[:640,:640];world=np.stack(((xx+.5)*14.26/640-7.13,7.13-(yy+.5)*14.26/640,np.zeros_like(xx)),-1)
    config=json.loads(args.calibration.read_text(encoding='utf-8'))
    projection_error=[]
    for slot,c in enumerate(sorted(config['camParams'],key=lambda x:x['camId'])):
        points=world[masks[slot]][::503].reshape(1,-1,3).astype(np.float64)
        e=np.array(c['Extrinsics']).reshape(3,4);rvec=cv2.Rodrigues(e[:,:3])[0]
        uv,_=cv2.fisheye.projectPoints(points,rvec,e[:,3],np.array(c['oriIntrinsics']).reshape(3,3),np.array(c['Distortion']))
        uv=uv.reshape(-1,2);uv=(uv+.5)*np.array([.5,1/3])-.5
        err=float(np.max(abs(uv-maps[slot][masks[slot]][::503])))
        assert err<.0001;projection_error.append(err)
    caps=[cv2.VideoCapture(str(p)) for p in src]
    if not all(c.isOpened() for c in caps):raise RuntimeError('Cannot open all sources')
    encoder=None;errfile=None;start=time.perf_counter();golden_results=[];encoded=0
    if not args.no_video:
        errfile=(out/'logs/ffmpeg.log').open('w',encoding='utf-8')
        encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','warning','-y','-f','rawvideo',
            '-pix_fmt','bgr24','-s','640x640','-r','25','-i','pipe:0','-an','-c:v','libx264',
            '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
            str(out/'avm_slam_full.part.mp4')],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errfile)
    try:
        for frame in range(len(pts)):
            working=[]
            for cap in caps:
                ok,image=cap.read()
                if not ok or image.shape[:2]!=(960,1280):raise RuntimeError(f'Decode failed at {frame}')
                working.append(cv2.resize(image,(640,320),interpolation=cv2.INTER_LINEAR))
            words=[pack565(im) for im in working]
            actual=replay.render(words)
            bgr=expand565(actual)
            if frame in GOLDEN:
                folder=out/'golden'/f'frame_{frame:04d}';folder.mkdir()
                expected=render(from_file,words,640,640)
                mismatch=int(np.count_nonzero(actual!=expected));assert mismatch==0
                for slot,w in enumerate(words):write_words(folder/f'slot_{slot}_{ROLES[slot]}',w)
                write_words(folder/'expected_rgb565',expected)
                cv2.imwrite(str(out/'preview'/f'frame_{frame:04d}_lut.png'),bgr)
                ref888=float_reference(working,maps,ca,cb,alpha,valid)
                ref565=float_reference([expand565(w) for w in words],maps,ca,cb,alpha,valid)
                cv2.imwrite(str(out/'preview'/f'frame_{frame:04d}_float.png'),ref888)
                row=dict(frame=frame,source_time_s=float(pts[frame]),integer_mismatches=mismatch,
                         rgb888_reference=metric(bgr,ref888,valid),
                         rgb565_packed_reference=metric(bgr,expand565(pack565(ref565)),valid))
                golden_results.append(row);save_json(out/'logs/golden_progress.json',golden_results)
                print('GOLDEN',frame,'PASS',flush=True)
            if encoder:
                payload=bgr.tobytes()
                for _ in range(int(repeats[frame])):encoder.stdin.write(payload)
                encoded+=int(repeats[frame])
            if frame%100==0:print(f'frame {frame}/{len(pts)} elapsed {time.perf_counter()-start:.1f}s',flush=True)
        for cap in caps:
            ok,_=cap.read()
            if ok:raise RuntimeError('Source longer than audited timestamps')
        if encoder:
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Video encoder failed; see log')
            (out/'avm_slam_full.part.mp4').rename(out/'avm_slam_full_640_lut.mp4')
    finally:
        for cap in caps:cap.release()
        if encoder and encoder.poll() is None:
            encoder.terminate();encoder.wait()
        if errfile:errfile.close()
    manifest=dict(format='AVM_96BIT_EXISTING_RTL',version=1,calibration_sha256=sha(args.calibration),
        output_size=[640,640],source_size=[640,320],raw_source_size=[1280,960],
        lut_bytes=(out/'lut/avm_slam_96bit_le.bin').stat().st_size,
        source_preprocessing='cv2.INTER_LINEAR pixel-center resize then truncating RGB565 pack',
        coordinate_frame='X right, Y forward, Z up; ground Z=0; metres',
        extent_m=dict(x=[-7.13,7.13],y=[-7.13,7.13]),meters_per_pixel=14.26/640,
        body_mask_m=dict(half_width=float(config['camParams'][0]['VisualStart']),half_length=float(config['camParams'][1]['VisualStart'])),
        projection='KB4 oriIntrinsics, row-major world-to-optical [R|t]',
        interpretation_status='geometry and image overlap validated; original author schema not confirmed',
        ignored_config_fields=['Intrinsics rectified K','skewingH','skewingW','isUndistort application flag'],
        vehicle_geometry_source='VisualStart controls body rectangle; no use of conflicting rearOverhang',
        cameras=[dict(slot=i,role=ROLES[i],source_file=str(src[i]),sha256=hashes[src[i].name]) for i in range(4)],
        camera_geometry=geometry,projection_vs_opencv_max_error_pixels=projection_error,
        coverage_outside_body=float(valid[~body].mean()),overlap_outside_body=float(mix[~body].mean()),
        body_pixels=int(body.sum()),source_frames=len(pts),source_pts_identical=True,
        exposure_sync='not independently verified',encoded_frames=encoded,encoded_fps=25 if encoder else None,
        video_duration_seconds=encoded/25 if encoder else None,elapsed_pc_seconds=time.perf_counter()-start,
        golden=golden_results,validation_scope='PC source decode, fixed-point mapping, binary/hex roundtrip; RTL reported separately; no hardware deployment')
    audit.update(calibration_status='PROJECTION_INTERPRETATION_VALIDATED_ON_PC',
                 camera_slot_assignment={str(i):NAMES[i] for i in range(4)},lut_generated=True)
    audit['output_contract']['lut_status']='generated; see manifest and separate RTL logs'
    save_json(out/'source/input_audit.json',audit)
    manifest['artifacts']={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
                           for p in out.rglob('*') if p.is_file()}
    save_json(out/'manifest.json',manifest)
    print('EXPORT COMPLETE',str(out),flush=True)


if __name__=='__main__':main()
