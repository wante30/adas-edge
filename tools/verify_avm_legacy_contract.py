"""Verify a delivered 480x640 legacy-compatible AVM-SLAM package."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from avm_fpga_fixed_model import read_lut, render
from avm_slam_lut import decode, encode


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    root=ap.parse_args().root
    p=json.loads((root/'4ch/parameters.json').read_text(encoding='utf-8'))
    assert (p['input_width'],p['input_height'],p['output_width'],p['output_height'])==(640,320,480,640)
    assert p['slot_camera_names']==['front','left','right','rear']
    b=(root/'4ch/lut/lut_96bit_le.bin').read_bytes()
    assert len(b)==3686400
    lines=(root/'4ch/lut/lut_readmemh96.hex').read_text().splitlines()
    assert len(lines)==307200 and all(len(line)==24 for line in lines)
    assert b''.join(int(line,16).to_bytes(12,'little') for line in lines)==b
    fields=read_lut(root/'4ch/lut/lut_96bit_le.bin',480,640)
    assert encode(decode(np.frombuffer(b,np.uint8))).tobytes()==b
    for side,active in [('a',fields['valid']!=0),('b',fields['blend']!=0)]:
        assert np.all(fields[f'cam_{side}'][~active]==7)
        assert np.all(fields[f'cam_{side}'][active]<4)
    for filename,key,dtype in [('valid_u8.bin','valid','u1'),('blend_u8.bin','blend','u1'),('alpha_u16_le.bin','alpha_q','<u2')]:
        assert np.array_equal(np.fromfile(root/'4ch/lut'/filename,dtype),fields[key])
    rows=[]
    for frame in (0,24,49,74,99):
        folder=root/'4ch/golden'/f'frame_{frame:03d}'
        images=[np.fromfile(folder/'inputs'/f'slot_{i}_{role}.bin','<u2').reshape(320,640) for i,role in enumerate(('front','left','right','rear'))]
        expected=np.fromfile(folder/'expected_rgb565.bin','<u2').reshape(640,480)
        mismatch=int(np.count_nonzero(render(fields,images,480,640)!=expected))
        assert mismatch==0
        rows.append(dict(frame=frame,pixels=307200,mismatches=mismatch))
        if frame==0:
            df=read_lut(root/'diagnostic/lut/lut_96bit_le.bin',480,640)
            diagnostic=np.fromfile(root/'diagnostic/expected_rgb565.bin','<u2').reshape(640,480)
            assert np.array_equal(render(df,images,480,640),diagnostic)
    if (root/'manifest.json').exists():
        for path,info in json.loads((root/'manifest.json').read_text(encoding='utf-8'))['artifacts'].items():
            resolved=(root/path).resolve()
            assert resolved.is_relative_to(root.resolve())
            data=resolved.read_bytes()
            assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256']
    print(json.dumps(dict(status='PASS',lut_bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),frames=rows),indent=2))


if __name__=='__main__':main()
