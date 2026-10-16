"""Synthetic contract tests; these are not real-video recognition accuracy."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.competition_console.navigation_reader import NavigationReader, classify_arrow, parse_distance


def arrow(kind):
    mask=np.zeros((52,40),np.uint8)
    if kind=='right':
        points=[(2,50),(2,27),(5,17),(23,4),(17,3),(36,1),(35,21),(29,13),(14,24),(10,30),(10,50)]
    elif kind=='left':
        points=[(1,12),(15,1),(15,8),(34,8),(34,50),(27,50),(27,16),(15,16),(15,24)]
    elif kind=='forward':
        points=[(20,1),(38,19),(25,19),(25,50),(15,50),(15,19),(2,19)]
    else:
        cv2.ellipse(mask,(20,18),(13,14),0,180,360,255,6)
        cv2.line(mask,(7,18),(7,36),255,6)
        cv2.line(mask,(33,18),(33,50),255,6)
        points=[(0,30),(14,30),(7,42)]
    cv2.fillPoly(mask,[np.asarray(points,np.int32)],255)
    y,x=np.where(mask>0)
    return mask[y.min():y.max()+1,x.min():x.max()+1]


def screen(kind='right'):
    image=np.full((540,960,3),160,np.uint8)
    mask=arrow(kind)
    # The calibrated primary-instruction position, distinct from map/lane arrows.
    h,w=mask.shape
    roi=image[60:60+h,40:40+w]
    roi[mask>0]=20
    cv2.putText(image,'893',(40+w+15,84),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,20),2)
    return image


class FakeOCR:
    def __init__(self,text='893',score=.99):
        self.calls=0;self.text=text;self.score=score
    def __call__(self,image,**kwargs):
        self.calls+=1
        assert kwargs=={'use_det':False,'use_cls':False}
        return SimpleNamespace(txts=(self.text,),scores=(self.score,))


@pytest.mark.parametrize('text,expected',[('893',893.),('518.',518.),('195米',195.),('1.2公里',1200.),('0米',0.),('12:35',None),('3分钟',None),('前方右转',None),('1.2',None),('12345',None),('123abc',None)])
def test_distance_is_instruction_number_not_clock_or_eta(text,expected):
    assert parse_distance(text)==expected


@pytest.mark.parametrize('kind',['right','left','forward','uturn'])
def test_arrow_shapes_are_distinguished(kind):
    actual,confidence=classify_arrow(arrow(kind))
    assert actual==kind and confidence>=.8


def test_filled_rectangle_is_not_instruction():
    assert classify_arrow(np.full((50,30),255,np.uint8))[0]=='none'


def test_observation_comes_from_pixels_not_frame_number():
    engine=FakeOCR('126')
    reader=NavigationReader(ocr=engine)
    first=reader.update(screen('left'),1,1)
    later=reader.update(screen('left'),9000,900_000_000)
    assert first.valid and later.valid and first.direction==later.direction=='left'
    assert first.distance_m==later.distance_m==126
    assert later.source_frame_id==9000 and later.timestamp_ns==900_000_000
    assert first.road_name==later.road_name==''


def test_pause_duplicate_does_not_reinfer_or_renew_timestamp():
    engine=FakeOCR();reader=NavigationReader(ocr=engine,ttl_ms=500)
    first=reader.update(screen(),10,1_000_000_000)
    assert first.valid
    paused=reader.update(screen('left'),10,1_100_000_000)
    assert paused==first and engine.calls==1
    expired=reader.update(screen(),10,1_500_000_000)
    assert not expired.valid and expired.direction=='none' and expired.distance_m is None
    assert expired.timestamp_ns==first.timestamp_ns and engine.calls==1


def test_non_navigation_view_clears_previous_immediately():
    reader=NavigationReader(ocr=FakeOCR())
    assert reader.update(screen(),1,0).valid
    side_view=np.full((540,960,3),80,np.uint8)
    cleared=reader.update(side_view,2,10_000_000)
    assert not cleared.valid and cleared.direction=='none' and cleared.distance_m is None
    assert not reader.current(100_000_000).valid


def test_map_and_lane_arrows_cannot_be_promoted_to_primary_instruction():
    image=np.full((540,960,3),160,np.uint8)
    cv2.arrowedLine(image,(500,400),(500,150),(0,0,0),20)
    cv2.arrowedLine(image,(60,190),(60,140),(0,0,0),5)
    engine=FakeOCR();result=NavigationReader(ocr=engine).update(image,0,0)
    assert not result.valid and engine.calls==0


@pytest.mark.parametrize('text,score',[('unknown',.99),('893',.2),('11:30',.99)])
def test_unreadable_distance_never_reuses_previous_value(text,score):
    engine=FakeOCR();reader=NavigationReader(ocr=engine)
    assert reader.update(screen(),1,0).valid
    engine.text=text;engine.score=score
    result=reader.update(screen(),2,100_000_000)
    assert not result.valid and result.distance_m is None and result.direction=='none'


def test_wrong_shape_and_session_replay():
    engine=FakeOCR();reader=NavigationReader(ocr=engine)
    assert reader.update(screen(),100,1000).valid
    assert not reader.update(np.zeros((30,30,3),np.uint8),101,1100).valid
    replay=reader.update(screen('left'),0,0)
    assert replay.valid and replay.direction=='left' and replay.source_frame_id==0
    reader.reset()
    assert not reader.current(0).valid


@pytest.mark.parametrize('dx,dy',[(18,-25),(-25,40),(-28,0)])
def test_photographed_card_position_can_move_without_a_time_lookup(dx,dy):
    image=cv2.warpAffine(screen(),np.float32([[1,0,dx],[0,1,dy]]),(960,540),
                         borderValue=(160,160,160))
    observation=NavigationReader(ocr=FakeOCR()).update(image,123,999)
    assert observation.valid and observation.direction=='right'


def test_digit_component_crop_excludes_small_unit_and_road_text():
    image=screen()
    # Same-row small text resembles extra digits to OCR, the real-video failure
    # behind 14 -> 1440 and 113 -> 1134. It must not enter the OCR line.
    cv2.putText(image,'40',(144,83),cv2.FONT_HERSHEY_SIMPLEX,.35,(20,20,20),1)
    found=NavigationReader._arrow(image)
    assert found is not None
    number_crop=found[-1]
    assert number_crop.shape[1] < 60


def test_clipped_leading_digit_is_rejected_instead_of_reading_suffix():
    image=np.full((540,960,3),160,np.uint8)
    # Numeric ROI ends at y=67 for this compact arrow. First glyph is clipped,
    # second is complete; reading only the second would turn 24 into 4.
    image[48:72,64:68]=20
    image[49:64,76:84]=20
    assert NavigationReader._digits(image,(30,40,25,20)) is None


def test_lower_road_name_row_cannot_truncate_last_digit():
    image=screen()
    before=NavigationReader._arrow(image)[-1]
    cv2.putText(image,'AAA',(112,105),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),2)
    after=NavigationReader._arrow(image)[-1]
    assert after.shape==before.shape
