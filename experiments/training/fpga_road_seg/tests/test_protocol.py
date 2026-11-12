import copy
import numpy as np
import pytest
from experiments.training.fpga_road_seg.mask_protocol import encode,decode,OfflineMaskProvider,transfer_budget
from experiments.training.fpga_road_seg.offline_q4 import crop_q4,validate_detections,fuse


@pytest.mark.parametrize('encoding',['bitpack-le','rle-u32','probability-u8'])
@pytest.mark.parametrize('kind',['random','zeros','ones'])
def test_transport_roundtrip_with_nonbyte_aligned_width(encoding,kind):
    rng=np.random.default_rng(15)
    mask=rng.random((4,7,13)) if kind=='random' else np.full((4,7,13),kind=='ones',dtype=float)
    packet=encode(mask,frame_id=7,timestamp_ns=100,encoding=encoding)
    result=decode(packet)
    if encoding=='probability-u8':
        assert np.max(np.abs(result-mask))<=.5/255+1e-7
    else: np.testing.assert_array_equal(result,mask>=.5)


def test_reject_invalid_crc_geometry_source_confidence():
    packet=encode(np.zeros((4,8,8)),frame_id=1,timestamp_ns=100)
    for key,value in [('source_id','mosaic'),('shape',[4,99999999,99]),('crc32','deadbeef'),
                      ('confidence',[float('nan')]*4),('frame_id',-1),('hardware',True),
                      ('class_names',list(reversed(packet['class_names'])))]:
        bad=copy.deepcopy(packet); bad[key]=value
        with pytest.raises(ValueError): decode(bad)


def test_provider_rejects_stale_future_frame_epoch_shape_and_explicit_invalid():
    packet=encode(np.zeros((4,8,16)),frame_id=5,timestamp_ns=1_000_000_000,stream_epoch=2)
    provider=OfflineMaskProvider(packet,max_age_ms=100)
    request=dict(frame_id=5,timestamp_ns=1_000_000_000,now_ns=1_050_000_000,stream_epoch=2,shape=[8,16])
    assert provider.get(**request)[1]['valid']
    for changes in (dict(frame_id=6),dict(now_ns=1_200_000_000),dict(now_ns=900_000_000),
                    dict(timestamp_ns=1_000_000_001),dict(stream_epoch=3),dict(shape=[16,8])):
        assert provider.get(**(request|changes))[0] is None
    invalid=packet|{'valid':False}
    assert OfflineMaskProvider(invalid).get(**request)[0] is None


def test_q4_crop_and_same_frame_object_fusion():
    image=np.zeros((20,40,3),np.uint8); image[10:,20:]=250
    assert crop_q4(image,'mosaic_2x2').shape==(10,20,3)
    assert crop_q4(image,'mosaic_2x2').min()==250
    detections=dict(frame_id=1,timestamp_ns=100,stream_epoch=0,shape=[10,20],source_id='Q4',coordinate_space='q4_native',
                    detections=[dict(bbox=[2,2,8,9],label='person')])
    rows=validate_detections(detections,frame_id=1,timestamp_ns=100,shape=[10,20])
    mask=np.zeros((4,10,20)); mask[0,7:,2:8]=1
    assert fuse(mask,rows)[0]['mask_footprint_overlap']['drivable_area']==1
    with pytest.raises(ValueError,match='mismatch'):
        validate_detections(detections,frame_id=2,timestamp_ns=100,shape=[10,20])


def test_bitpack_bandwidth_and_rle_worst_case_not_assumed_compressed():
    budget=transfer_budget()
    assert budget['payload_only']['bitpack_le_bytes']==30720
    assert budget['bytes_per_second']['bitpack_le_bytes']==921600
    assert budget['payload_only']['probability_u8_bytes']==245760
    assert budget['payload_only']['rle_worst_case_bytes']>245760
