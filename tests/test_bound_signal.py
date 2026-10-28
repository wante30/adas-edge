import cv2
import numpy as np

from competition_ai.bound_signal import BoundSignal, classify_horizontal_head
from competition_ai.scenario import Light


def housing(colour, dx=0):
    image = np.full((160,320,3), 225, np.uint8)
    cv2.rectangle(image, (60+dx,50), (210+dx,100), (20,20,20), -1)
    cv2.rectangle(image, (60+dx,50), (210+dx,100), (160,160,160), 2)
    for i in range(3):
        cv2.circle(image, (85+dx+i*48,75), 13, (55,55,55), -1)
    i = 0 if colour == 'red' else 2
    cv2.circle(image, (85+dx+i*48,75), 12, (0,0,255) if i == 0 else (0,255,0), -1)
    return image


def test_green_confirmation_uses_distinct_frames_and_bound_housing_motion():
    signal = BoundSignal(housing('green'), [57,47,214,104])
    assert signal.update(housing('green'), 0, 0).light == Light.UNKNOWN
    assert signal.update(housing('green'), 0, 0).light == Light.UNKNOWN
    result = signal.update(housing('green',3), 1, 100_000_000)
    assert result.light == Light.GREEN and result.associated
    assert abs(result.bbox[0]-60) < 2
    assert not signal.current(500_000_000).associated


def test_red_immediate_and_replay_jump_cancels_binding():
    signal = BoundSignal(housing('red'), [57,47,214,104])
    assert signal.update(housing('red'), 10, 0).light == Light.RED
    result = signal.update(housing('red'), 0, 0)
    assert result.light == Light.UNKNOWN and not result.associated


def test_background_replacement_cannot_turn_an_old_roi_green():
    signal = BoundSignal(housing('red'), [57,47,214,104])
    signal.update(housing('red'), 0, 0)
    leaves = np.full((160,320,3), (20,170,20), np.uint8)
    result = signal.update(leaves, 1, 100_000_000)
    assert result.light == Light.UNKNOWN and not result.associated


def test_tracking_only_does_not_refresh_colour_source_time():
    signal = BoundSignal(housing('red'), [57,47,214,104])
    signal.update(housing('red'), 0, 0)
    signal.update(housing('red',2), 1, 100_000_000, classify=False)
    assert signal.current(100_000_000).timestamp_ns == 0


def test_green_background_around_red_head_does_not_win_colour_vote():
    frame = housing('red')
    frame[:45] = (0,200,0)
    frame[105:] = (0,200,0)
    assert classify_horizontal_head(frame,[57,47,214,104]) == Light.RED
    # A whole patch of leaves has no bound dark housing.
    assert classify_horizontal_head(np.full_like(frame,(0,200,0)),[57,47,214,104]) == Light.UNKNOWN


def test_bound_single_lamp_housing_confirms_centered_red_light():
    frame = np.full((120,240,3),225,np.uint8)
    cv2.rectangle(frame,(80,42),(160,78),(20,20,20),-1)
    cv2.rectangle(frame,(80,42),(160,78),(160,160,160),2)
    cv2.circle(frame,(120,60),10,(0,0,255),-1)

    assert classify_horizontal_head(frame,[76,38,164,82]) == Light.RED


def test_zoom_does_not_accumulate_housing_aspect_ratio_drift():
    original = housing('green')
    signal = BoundSignal(original,[57,47,214,104])
    for fid in range(35):
        zoom = 1+fid*.005
        transform = np.float32([[zoom,0,(1-zoom)*135],[0,zoom,(1-zoom)*75]])
        frame = cv2.warpAffine(original,transform,(320,160),borderValue=(225,225,225))
        result = signal.update(frame,fid,fid*30_000_000)
    assert result.light == Light.GREEN and result.associated
    x1,y1,x2,y2 = result.bbox
    assert abs((x2-x1)/(y2-y1)-157/57) < .06
