import cv2
import numpy as np
import pytest

from competition_ai.hazard_events import HazardEvents, paired_red_lamps
from competition_ai.image_motion import ImageMotion

STILL = ImageMotion(valid=True, inlier_ratio=1., reason='SYNTHETIC_TEST')
FRAME = np.zeros((180, 320, 3), np.uint8)


def run_vehicle(sizes, *, motion=STILL, center=160):
    engine = HazardEvents()
    results = []
    for i, size in enumerate(sizes):
        box = [center-size/2, 128-size*.65, center+size/2, 128]
        results.append(engine.update(FRAME, [{'class_name': 'car', 'confidence': .9, 'bbox': box}],
                                     i, i*100_000_000, motion=motion))
    return engine, results


def test_consistent_lead_expansion_warns_but_does_not_claim_measured_sudden_braking():
    engine, observations = run_vehicle(np.linspace(28, 66, 18))
    assert any(o.obstacle_hazard for o in observations)
    assert observations[-1].sudden_braking.startswith('UNKNOWN')
    assert len({o.lead_id for o in observations}) == 1
    assert not engine.current(2_000_000_000).obstacle_hazard


@pytest.mark.parametrize('sizes', [np.full(18, 46), np.linspace(66, 28, 18),
                                  [40, 42, 39, 46, 37, 40, 53, 40, 38, 42, 49, 40, 41, 44, 40, 39, 42, 40]])
def test_normal_following_receding_and_detector_jitter_do_not_warn(sizes):
    _, observations = run_vehicle(sizes)
    assert not any(o.obstacle_hazard for o in observations)


def test_nearby_off_path_vehicle_and_camera_pan_do_not_warn():
    _, off_path = run_vehicle(np.linspace(28, 66, 18), center=55)
    _, roadside = run_vehicle(np.linspace(28, 66, 18), center=200)
    _, turning = run_vehicle(np.linspace(28, 66, 18), motion=ImageMotion(valid=True, dx=.25))
    assert not any(o.obstacle_hazard for o in off_path+roadside+turning)


def test_uniform_scene_zoom_does_not_become_a_lead_collision_warning():
    engine = HazardEvents()
    for i in range(18):
        size = 28*np.exp(.5*i*.1)
        box = [160-size/2, 128-size*.65, 160+size/2, 128]
        observed = engine.update(FRAME, [{'class_name':'car','confidence':.9,'bbox':box}],
                                 i,i*100_000_000,motion=ImageMotion(valid=True,scale_rate=.5))
        assert not observed.obstacle_hazard


def test_pedestrian_crossing_path_warns_without_using_vehicle_red_signal():
    engine = HazardEvents()
    observations = []
    for i in range(18):
        x = 125+i*2
        observations.append(engine.update(FRAME, [{'class_name': 'person', 'confidence': .9,
                                                   'bbox': [x-8, 88, x+8, 138]}],
                                          i, i*100_000_000, motion=STILL))
    assert any(o.pedestrian_hazard for o in observations)
    assert all(o.pedestrian_red_violation.startswith('UNKNOWN') for o in observations)


def test_pedestrian_camera_displacement_and_rider_are_not_crossing_on_foot():
    engine = HazardEvents()
    for i in range(18):
        x = 120+i*2
        observation = engine.update(FRAME, [{'class_name': 'person', 'confidence': .9,
                                             'bbox': [x-8, 88, x+8, 138]}],
                                    i, i*100_000_000, motion=ImageMotion(valid=True, dx=2/320/.1))
        assert not observation.pedestrian_hazard
    engine.reset()
    for i in range(18):
        x = 125+i*2
        observation = engine.update(FRAME, [{'class_name': 'person', 'confidence': .9, 'bbox': [x-8, 88, x+8, 138]},
                                             {'class_name': 'motorcycle', 'confidence': .9, 'bbox': [x-12, 116, x+12, 155]}],
                                    i, i*100_000_000, motion=STILL)
        assert not observation.pedestrian_hazard


def test_pause_gap_and_restart_do_not_reconfirm_or_reuse_old_events():
    engine, observations = run_vehicle(np.linspace(28, 66, 18))
    before = len(engine.timings)
    engine.update(FRAME, [], 17, 1_700_000_000, motion=STILL)
    assert len(engine.timings) == before
    assert not engine.update(FRAME, [], 18, 3_000_000_000, motion=STILL).events
    assert not engine.update(FRAME, [], 0, 0, motion=STILL).events
    assert not engine.tracker.tracks


def test_stationary_person_and_sidewalk_walker_do_not_warn():
    for positions in ([160]*18, range(30, 48)):
        engine = HazardEvents()
        for i, x in enumerate(positions):
            observation = engine.update(FRAME, [{'class_name': 'person', 'confidence': .9,
                                                 'bbox': [x-8, 88, x+8, 138]}],
                                        i, i*100_000_000, motion=STILL)
            assert not observation.pedestrian_hazard


def test_red_lamp_pixels_alone_never_trigger_collision_or_confirm_acute_braking():
    engine = HazardEvents()
    box = [110, 70, 210, 134]
    for i in range(18):
        frame = FRAME.copy()
        if i > 8:
            cv2.rectangle(frame, (115, 100), (135, 113), (0, 0, 255), -1)
            cv2.rectangle(frame, (185, 100), (205, 113), (0, 0, 255), -1)
        observation = engine.update(frame, [{'class_name': 'car', 'confidence': .9, 'bbox': box}],
                                    i, i*100_000_000, motion=STILL)
        assert not observation.obstacle_hazard
        assert not any(e['kind'] == 'SUSPECTED_BRAKING_WITH_CLOSING' for e in observation.events)
    assert paired_red_lamps(frame, box) > .025
    assert observation.tracks[0]['brake_lamp_rise_candidate']
    assert any(e['kind']=='LEAD_BRAKE_LAMP_RISE' for e in observation.events)


def test_invalid_path_and_unconfirmed_background_are_explicit():
    with pytest.raises(ValueError):
        HazardEvents(((0, 0), (1, 1), (0, 1), (1, 0)))
    _, observations = run_vehicle(np.linspace(28, 66, 18), motion=ImageMotion())
    assert not any(o.events for o in observations)
    assert observations[-1].reason == 'camera_motion_or_unconfirmed_background'
