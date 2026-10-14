"""Hardware format, boundary and independent replay equivalence checks."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from avm_slam_lut import FIELDS,encode,decode,from_maps,Replay,pack565,expand565
from avm_fpga_fixed_model import render


def fields():
    rng=np.random.default_rng(5626)
    n=33*19
    f={k:np.zeros(n,np.uint32) for k in FIELDS}
    for s in ['a','b']:
        f[f'addr_{s}']=rng.integers(0,319,n,dtype=np.uint32)*640+rng.integers(0,639,n,dtype=np.uint32)
        f[f'cam_{s}']=rng.integers(0,4,n,dtype=np.uint32)
        for axis in ['fx','fy']:f[f'{axis}_{s}']=rng.integers(0,16,n,dtype=np.uint32)
    f['valid'][:]=1;f['blend'][:]=1;f['alpha_q']=rng.integers(0,65,n,dtype=np.uint32)
    f['valid'][::17]=0;f['blend'][::17]=0
    return f


def test_byte_layout_against_independent_python_integer():
    f=fields();raw=encode(f)
    for i,row in enumerate(raw):
        word=sum(int(f[k][i])<<offset for k,(offset,_) in FIELDS.items())
        assert row.tobytes()==word.to_bytes(12,'little')
    decoded=decode(raw)
    assert all(np.array_equal(decoded[k],f[k]) for k in f)


def test_fast_replay_matches_existing_integer_reference():
    f=decode(encode(fields()));rng=np.random.default_rng(3566)
    images=[rng.integers(0,65536,(320,640),dtype=np.uint16) for _ in range(4)]
    assert np.array_equal(Replay(f,(33,19)).render(images),render(f,images,33,19))


def test_invalid_and_unused_samples_never_address_source_memory():
    f=fields()
    for s in ['a','b']:
        f[f'cam_{s}'][::17]=7;f[f'addr_{s}'][::17]=(1<<18)-1
    f['blend'][1]=0;f['alpha_q'][1]=64;f['cam_b'][1]=7;f['addr_b'][1]=(1<<18)-1
    f=decode(encode(f))
    images=[np.full((320,640),i*10000,np.uint16) for i in range(4)]
    assert np.array_equal(Replay(f,(33,19)).render(images),render(f,images,33,19))


@pytest.mark.parametrize('field,value',[('alpha_q',65),('fx_a',16),('cam_a',4),('addr_a',639),('addr_a',319*640)])
def test_reject_hardware_unsafe_entry(field,value):
    f=fields();f[field][1]=value
    with pytest.raises(ValueError):encode(f)


def test_reject_reserved_bit():
    raw=encode(fields());raw[1,11]=1
    with pytest.raises(ValueError):decode(raw)


def test_rounding_carry_cannot_read_past_source_row():
    maps=np.full((4,1,1,2),638.99,np.float32);maps[...,1]=10
    with pytest.raises(ValueError):from_maps(maps,np.zeros((1,1),int),np.ones((1,1),int),np.ones((1,1)),np.ones((1,1),bool),np.zeros((1,1),bool))


def test_rgb565_roundtrip_all_codes():
    words=np.arange(65536,dtype=np.uint16)
    assert np.array_equal(pack565(expand565(words)),words)
