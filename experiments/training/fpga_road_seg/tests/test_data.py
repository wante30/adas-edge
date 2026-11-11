import json
import numpy as np
import pytest
from PIL import Image
from experiments.training.fpga_road_seg.dataset_adapter import (
    bdd_targets, vistas_targets, poly_points, raster, letterbox, restore,
    validate_manifest, convert_index, RoadDataset, select_frame,
)


def poly(vertices, closed=False, types=None):
    return dict(vertices=vertices, closed=closed, types=types or 'L'*len(vertices))


def test_bdd_real_schema_missing_crosswalk_and_unknown_style(tmp_path):
    a=np.full((20,30),2,np.uint8); a[10:]=0; a[0,0]=255
    dp=tmp_path/'drive.png'; Image.fromarray(a).save(dp)
    frame=dict(labels=[
        dict(category='lane/single white',attributes={'laneStyle':'solid'},poly2d=[poly([[5,5],[5,18]])]),
        dict(category='single yellow',attributes={'laneStyle':'dashed'},poly2d=[poly([[15,6],[15,16]])]),
        dict(category='single other',attributes={},poly2d=[poly([[25,5],[25,16]])]),
        dict(category='crosswalk',attributes={'laneStyle':'solid'},poly2d=[poly([[0,10],[29,10]])]),
    ])
    out=bdd_targets(a.shape,drivable=dp,lane_frame=frame,line_width=2)
    y,v=out['target'],out['valid']
    assert y[0,14,5] == y[1,14,5] == 1  # overlapping channels preserved
    assert y[2,10,15] == 1 and y[1,10,15] == 0
    assert v[0,0,0] == 0 and v[3].sum() == y[3].sum() > 0
    assert np.array_equal(v[3], y[3])  # weak positive-only curves, not dense labels
    assert v[1:3,10,25].sum() == 0 and out['lane'][10,25] == 1
    assert y[1:3,10,20].sum() == 0  # sparse crosswalk not a solid lane
    missing=bdd_targets(a.shape)
    assert missing['valid'].sum() == missing['lane_valid'].sum() == 0
    assert bdd_targets(a.shape,lane_frame={'labels':[]})['valid'][1].all()
    with pytest.raises(ValueError):
        bdd_targets(a.shape,lane_frame={})


def test_bezier_not_control_polygon_and_width_sensitivity():
    curve=poly([[2,10],[2,0],[18,0],[18,10]],types='LCCC')
    pts=poly_points(curve)
    assert len(pts)>4 and pts[:,1].min()>0
    thin=raster(curve,(24,24),2); thick=raster(curve,(24,24),6)
    assert thick.sum()>thin.sum()
    leading_controls=poly([[2,10],[2,0],[18,0],[18,10]],types='CCCL')
    assert len(poly_points(leading_controls)) > 4
    with pytest.raises(ValueError):
        poly_points(poly([[0,0],[1,1]],types='LX'))


def test_historical_bdd_objects_style_flat_bezier_and_positive_only_png(tmp_path):
    legacy = {'objects': [
        {'category': 'area/drivable', 'poly2d': [[[1, 1, 'L'], [10, 1, 'L'], [10, 10, 'L'], [1, 10, 'L']]]},
        {'category': 'lane/single white', 'attributes': {'style': 'solid'},
         'poly2d': [[2, 2, 'L'], [2, 4, 'C'], [8, 4, 'C'], [8, 8, 'C']]},
        {'category': 'lane/single yellow', 'attributes': {'style': 'dashed'},
         'poly2d': [[[12, 2, 'L'], [12, 12, 'L']]]},
    ]}
    out = bdd_targets((16, 16), drivable_frame=legacy, lane_frame=legacy, line_width=2)
    assert out['target'][0, 5, 5] == 1 and out['valid'][0].all()
    assert out['target'][1].sum() > 0 and out['target'][2].sum() > 0
    missing = bdd_targets((16,16), drivable_frame={'objects':[]}, lane_frame={'objects':[]})
    assert missing['valid'].sum() == missing['lane_valid'].sum() == 0

    # Historical PNG renderer used 255 as background; only encoded line pixels
    # are valid so unlabeled background can never become a negative example.
    png = np.full((16, 16), 255, np.uint8)
    png[3:7, 3] = 6  # category single white, style solid
    path = tmp_path / 'legacy-lane.png'
    Image.fromarray(png).save(path)
    decoded = bdd_targets((16, 16), lane_png=path)
    assert decoded['target'][1].sum() == 4
    assert decoded['valid'][1].sum() == 4
    assert decoded['valid'][1, 0, 0] == 0


def test_historical_bdd_multiframe_json_is_matched_by_image_name():
    rows = [{'name': 'a.jpg', 'objects': []}, {'name': 'b.jpg', 'objects': []}]
    assert select_frame(rows, 'b.jpg') is rows[1]
    with pytest.raises(ValueError, match='expected one BDD frame'):
        select_frame(rows, 'missing.jpg')


def test_official_single_image_json_wrapper_selects_its_frame():
    data = {'name': 'video-frame', 'frames': [{'objects': []}]}
    assert select_frame(data, 'video-frame.jpg') is data['frames'][0]


def test_vistas_uses_config_names_void_and_missing_styles(tmp_path):
    cfg=tmp_path/'config.json'
    cfg.write_text(json.dumps({'labels':[{'name':'void--unlabeled'},{'name':'marking--crosswalk-zebra'},{'name':'construction--flat--road'}]}),encoding='utf-8')
    a=np.array([[0,1,2],[2,1,2]],np.uint8); lp=tmp_path/'label.png'; Image.fromarray(a).save(lp)
    out=vistas_targets(a.shape,lp,cfg,{'crosswalk':['marking--crosswalk-zebra']})
    assert out['target'][3].sum()==2 and out['valid'][3].sum()==5
    assert not out['valid'][:3].any()
    with pytest.raises(ValueError):
        vistas_targets(a.shape,lp,cfg,{'solid_lane':['construction--flat--road']})


def test_letterbox_padding_is_ignored_and_restore_aligns():
    im=np.zeros((10,20,3),np.uint8); y=np.ones((4,10,20),np.uint8)
    _,t,v,g=letterbox(im,y,y,size=(40,40))
    assert t.shape==(4,40,40) and v[:,0].sum()==0
    restored=restore(t.numpy(),g)
    np.testing.assert_array_equal(restored,y)


def test_group_leakage_and_cross_split_duplicate_content_rejected(tmp_path):
    rows=[dict(id='a',image='a.png',masks='a.npz',split='train',group_id='video1',license='terms'),
          dict(id='b',image='b.png',masks='b.npz',split='test',group_id='video1',license='terms')]
    with pytest.raises(ValueError,match='leakage'):
        validate_manifest(rows,check_files=False)
    for r in rows:
        r['image']=str(tmp_path/(r['id']+'.png')); r['masks']=str(tmp_path/(r['id']+'.npz'))
        Image.new('RGB',(8,8)).save(r['image']); np.savez(r['masks'],a=[1])
    rows[1]['group_id']='video2'
    rows[1]['split']='train'
    validate_manifest(rows)
    rows[1]['split']='test'
    with pytest.raises(ValueError,match='content'):
        validate_manifest(rows)


def test_convert_and_load_native_bdd_masks(tmp_path, monkeypatch):
    im=tmp_path/'frame.jpg'; Image.new('RGB',(30,20)).save(im)
    dp=tmp_path/'drivable.png'; Image.fromarray(np.zeros((20,30),np.uint8)).save(dp)
    lane=tmp_path/'lane.json'; lane.write_text(json.dumps({'labels':[]}),encoding='utf-8')
    index=tmp_path/'index.json'
    index.write_text(json.dumps([dict(id='frame',image='frame.jpg',source='bdd100k',drivable_png='drivable.png',lane_json='lane.json',split='train',group_id='video',license='BDD terms')]),encoding='utf-8')
    mp=convert_index(index,tmp_path/'converted')
    x,y,v=RoadDataset(mp,'train',(32,32))[0]
    assert x.shape==(3,32,32) and y.shape==(4,32,32)
    assert v[3].sum()==0 and y[0].sum()>0
    monkeypatch.setenv('BDD_SEG_CACHE', str(tmp_path/'tensor-cache'))
    for baseline in (False, True):
        ds=RoadDataset(mp,'train',(32,32),baseline,('crosswalk',))
        first=ds[0]
        cached=ds[0]
        for a,b in zip(first,cached):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
    with pytest.raises(ValueError, match='line width'):
        convert_index(index,tmp_path/'converted',line_width=4,resume=True)
