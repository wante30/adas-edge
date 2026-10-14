"""Verify a delivered AVM-SLAM package without access to the original videos."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from avm_slam_lut import decode,Replay,expand565
from avm_fpga_fixed_model import render


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for relative,info in manifest['artifacts'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError('Artifact path escapes package')
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if path.stat().st_size!=info['bytes'] or digest!=info['sha256']:
            raise ValueError(f'Artifact mismatch: {relative}')
    raw=np.fromfile(root/'lut/avm_slam_96bit_le.bin',np.uint8)
    assert raw.size==640*640*12
    fields=decode(raw.reshape(-1,12));replay=Replay(fields)
    lines=(root/'lut/avm_slam_96bit.hex').read_text().splitlines()
    assert len(lines)==640*640
    assert b''.join(int(line,16).to_bytes(12,'little') for line in lines)==raw.tobytes()
    rows=[]
    for entry in manifest['golden']:
        frame=entry['frame'];folder=root/'golden'/f'frame_{frame:04d}'
        images=[]
        for camera in manifest['cameras']:
            words=np.fromfile(folder/f'slot_{camera["slot"]}_{camera["role"]}.bin',dtype='<u2')
            assert words.size==640*320
            images.append(words.reshape(320,640))
        expected=np.fromfile(folder/'expected_rgb565.bin',dtype='<u2').reshape(640,640)
        actual=replay.render(images);independent=render(fields,images,640,640)
        mismatch=int(np.count_nonzero(actual!=expected))
        assert mismatch==0 and np.array_equal(actual,independent)
        preview=cv2.imread(str(root/'preview'/f'frame_{frame:04d}_lut.png'))
        assert np.array_equal(preview,expand565(expected))
        rows.append(dict(frame=frame,pixels=expected.size,mismatches=mismatch))
    print(json.dumps(dict(status='PASS',artifact_hashes=len(manifest['artifacts']),
                          lut_entries=640*640,golden=rows),indent=2))


if __name__=='__main__':main()
