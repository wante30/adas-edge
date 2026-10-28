"""Regression checks for the old FPGA ABI rather than just a rotated image."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from avm_slam_lut import FIELDS, Replay
from avm_fpga_fixed_model import render
from export_avm_legacy_contract import with_legacy_slots_and_sentinels, source_inspection_fields


def sample():
    fields={key:np.zeros(8,np.uint32) for key in FIELDS}
    fields['cam_a']=np.tile(np.arange(4,dtype=np.uint32),2)
    fields['cam_b']=np.tile(np.arange(4,dtype=np.uint32)[::-1],2)
    fields['valid'][:6]=1
    fields['blend'][:4]=1
    fields['alpha_q'][:4]=[32,40,50,63]
    fields['alpha_q'][4:6]=64
    for s in ('a','b'):
        fields[f'addr_{s}'][:]=640*100+200
        fields[f'fx_{s}'][:]=3
        fields[f'fy_{s}'][:]=11
    return fields


def test_remap_is_simultaneous_not_cascading():
    changed=with_legacy_slots_and_sentinels(sample())
    assert changed['cam_a'].tolist()==[1,0,2,3,1,0,7,7]
    assert changed['cam_b'].tolist()==[3,2,0,1,7,7,7,7]


def test_unused_samples_use_old_sentinel_and_zero_payload():
    changed=with_legacy_slots_and_sentinels(sample())
    for s,active in [('a',changed['valid']!=0),('b',changed['blend']!=0)]:
        assert np.all(changed[f'cam_{s}'][~active]==7)
        for key in ('addr','fx','fy'):assert np.all(changed[f'{key}_{s}'][~active]==0)
    assert np.all(changed['alpha_q'][changed['valid']==0]==0)


def test_slot_remap_keeps_mixed_pixel_values_identical():
    original=sample();adapted=with_legacy_slots_and_sentinels(original)
    rng=np.random.default_rng(20261016)
    images=[rng.integers(0,65536,(320,640),dtype=np.uint16) for _ in range(4)]
    swapped=[images[1],images[0],images[2],images[3]]
    assert np.array_equal(Replay(original,(4,2)).render(images),render(adapted,swapped,4,2))


def test_diagnostic_bands_show_all_four_slots_without_rotation():
    fields=source_inspection_fields()
    colors=[0xf800,0x07e0,0x001f,0xffff]
    images=[np.full((320,640),color,np.uint16) for color in colors]
    result=render(fields,images,480,640)
    for slot,color in enumerate(colors):
        assert np.all(result[slot*160:(slot+1)*160,80:400]==color)
    assert np.all(result[:,:80]==0) and np.all(result[:,400:]==0)
    # Top-left and bottom-right thumbnail samples must be in that order.
    for slot in range(4):
        a=slot*160*480+80;b=(slot*160+159)*480+399
        assert fields['addr_a'][a]==0
        assert fields['addr_a'][b]==318*640+638
