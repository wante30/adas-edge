"""Supervision masking and forgetting guard regression checks (no training)."""
import json
import numpy as np
from PIL import Image

from experiments.training.fpga_road_seg.dataset_adapter import RoadDataset
from experiments.training.fpga_road_seg.train import selection_score
from experiments.training.fpga_road_seg.quantize_stage2 import balanced_calibration, maximum_drop


def test_row_channel_disable_applies_to_native_and_shared_cache(tmp_path, monkeypatch):
    image = tmp_path/'road.png'
    Image.new('RGB', (16, 8)).save(image)
    target = np.ones((4, 8, 16), np.uint8)
    masks = tmp_path/'road.npz'
    np.savez(masks, target=target, valid=target, lane=target[1], lane_valid=target[1])
    row = dict(id='bdd', source='bdd100k', group_id='road', license='test', split='train',
               image=str(image), masks=str(masks), disabled_channels=['crosswalk'])
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps(row)+'\n', encoding='utf-8')
    ds = RoadDataset(manifest, 'train', (16,8))
    assert ds[0][2][3].sum() == 0
    monkeypatch.setenv('BDD_SEG_CACHE', str(tmp_path/'cache'))
    ds = RoadDataset(manifest, 'train', (16,8))
    first, cached = ds[0], ds[0]
    assert first[2][:3].sum() > 0 and cached[2][3].sum() == 0
    np.testing.assert_array_equal(first[2], cached[2])
    # Cache itself must retain raw supervision for other experiments.
    ds.rows[0]['disabled_channels'] = ['drivable_area']
    assert ds[0][2][0].sum() == 0 and ds[0][2][3].sum() > 0


def test_high_crosswalk_score_cannot_hide_bdd_task_regression():
    baseline = dict(bdd100k=dict(classes=dict(drivable_area=dict(iou=.55),
        solid_lane=dict(f1=.56), dashed_lane=dict(f1=.45))), rlmd=dict(classes=dict(crosswalk=dict(iou=.01))))
    current = json.loads(json.dumps(baseline))
    current['rlmd']['classes']['crosswalk']['iou'] = .8
    current['bdd100k']['classes']['dashed_lane']['f1'] = .40
    assert selection_score(current, baseline, .025)[1] is False
    current['bdd100k']['classes']['dashed_lane']['f1'] = .44
    assert selection_score(current, baseline, .025)[1] is True


def test_calibration_uses_balanced_train_domains_and_crosswalk_negatives():
    rows=[]
    for source in ('bdd100k','rlmd'):
        for split in ('train','val','test'):
            for i in range(10):
                rows.append(dict(id=f'{source}-{split}-{i}',source=source,split=split,positive_pixels=[0,0,0,i%2]))
    selected=balanced_calibration(rows,per_domain=8)
    assert len(selected)==16 and all(r['split']=='train' for r in selected)
    assert sum(r['source']=='bdd100k' for r in selected)==8
    rlmd=[r for r in selected if r['source']=='rlmd']
    assert sum(r['positive_pixels'][3]>0 for r in rlmd)==4


def test_quantized_crosswalk_gain_cannot_hide_drivable_loss():
    original=dict(bdd100k=dict(classes=dict(drivable_area=dict(iou=.55),solid_lane=dict(f1=.56),dashed_lane=dict(f1=.45))),
                  rlmd=dict(classes=dict(crosswalk=dict(iou=.4,f1=.57,recall=.6))))
    candidate=json.loads(json.dumps(original))
    candidate['bdd100k']['classes']['drivable_area']['iou']=.45
    candidate['rlmd']['classes']['crosswalk']['iou']=.7
    assert maximum_drop(original,candidate)>.09
