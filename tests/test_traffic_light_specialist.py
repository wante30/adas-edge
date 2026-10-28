from __future__ import annotations

from tools.adas_hmi_sim.backend.traffic_light_specialist import _map_box


def test_specialist_rknn_box_maps_from_letterboxed_roi_to_full_frame() -> None:
    # 1920x648 ROI into a 512x192 canvas: scale=0.2666, vertical padding=10.
    mapped = _map_box(
        [128.0, 42.0, 256.0, 74.0],
        scale=512.0 / 1920.0,
        pad_x=0,
        pad_y=10,
        crop_w=1920,
        crop_h=648,
        roi_top=0.60,
        full_w=1920,
        full_h=1080,
    )
    assert mapped == [480.0, 120.0, 960.0, 240.0]


def test_specialist_box_is_clipped_to_source_frame() -> None:
    mapped = _map_box(
        [-100.0, -20.0, 900.0, 500.0],
        scale=1.0,
        pad_x=0,
        pad_y=0,
        crop_w=800,
        crop_h=400,
        roi_top=0.60,
        full_w=800,
        full_h=600,
    )
    assert mapped == [0.0, 0.0, 800.0, 400.0]


def test_focused_crop_box_restores_both_origin_offsets() -> None:
    mapped = _map_box([20, 10, 60, 30], scale=.5, pad_x=0, pad_y=0,
                      crop_w=600, crop_h=200, roi_top=.6, full_w=1000,
                      full_h=600, crop_origin=(200,150))
    assert mapped == [240.,170.,320.,210.]


def test_adaptive_roi_runs_only_one_inference_per_refresh():
    import numpy as np
    from tools.adas_hmi_sim.backend.traffic_light_specialist import TrafficLightSpecialist
    class FakeRunner:
        def __init__(self): self.calls=0
        def detect(self, frame): self.calls+=1; return []
    m=object.__new__(TrafficLightSpecialist)
    m.roi_mode='adaptive';m.roi_top=.6;m.confidence=.05
    m._pc_detector=FakeRunner();m.reset_roi()
    image=np.zeros((300,600,3),np.uint8)
    m.detect(image,frame_id=0);assert m.last_roi=='top60'
    m.detect(image,frame_id=10);assert m.last_roi=='focus'
    assert m._pc_detector.calls==2


def test_signal_scheduler_does_not_drift_at_30fps_three_hz_boundaries():
    from tools.adas_hmi_sim.backend.traffic_light_specialist import signal_refresh_due
    last=None;due=[]
    for fid in range(169):
        if signal_refresh_due(fid/30.,last,3.):
            due.append(fid);last=fid/30.
    assert due==list(range(0,169,10))
