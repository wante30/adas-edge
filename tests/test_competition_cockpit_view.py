"""The native view exposes playback and display tools, not control inputs."""
import numpy as np
import pytest
from competition_ai.bound_signal import SignalObservation
from competition_ai.scenario import Controls, Light, Mode, Perception
from competition_ai.scenario_policy import decide
from tools.competition_console.pipeline import FrontSnapshot
from tools.competition_console.view import NativeView


@pytest.fixture
def view():
    return NativeView()


def make_snapshot(main=None, signal=None, hazards=None):
    from competition_ai.hazard_events import HazardObservation

    return FrontSnapshot(main or [], [], signal or SignalObservation(), 12.0, None,
                         'UNBOUND', hazards or HazardObservation())


def render(view, mode=Mode.INTERSECTION, shape=(1080,1920,3), *, controls=None,
           snapshot=None, **kwargs):
    controls = controls or Controls(mode=mode)
    snapshot = snapshot or make_snapshot()
    decision = decide(controls, Perception(), 0)
    return view.render(np.zeros(shape,np.uint8),snapshot,controls,decision,
                       frame_id=0,fps=30,**kwargs)


@pytest.mark.parametrize('mode',list(Mode))
def test_only_pure_ui_buttons_are_clickable(view,mode):
    render(view,mode)
    assert [view.hit(x,41) for x in (538,703,868,1033)] == [None] * 4
    assert [view.hit(x,41) for x in (1214,1346,1497)] == ['o','f','c']
    assert [view.hit(x,685) for x in (537,1026,1117,1247,1437)] == [' ','.','r',None,'v']
    assert view.hit(201,842) is None
    assert view.hit(1414,770) is None
    assert view.hit(1500,70) is None


def test_legacy_control_hotspots_do_not_register_as_view_actions(view):
    render(view)
    for x, y in ((207,842),(371,842),(537,842),(829,842),(1000,842),
                 (1247,685),(1414,770)):
        assert view.hit(x,y) is None


def test_diagnostics_preserve_only_display_and_playback_actions(view):
    render(view,diagnostics=True)
    assert view.hit(1346,41)=='f'
    assert view.hit(868,41) is None
    assert view.hit(1247,685) is None
    assert view.hit(1026,685)=='.'


def test_control_mode_and_synthetic_decision_do_not_change_display(view):
    images = []
    for mode in Mode:
        controls=Controls(mode=mode)
        images.append(render(view,controls=controls))
    assert all(np.array_equal(images[0],image) for image in images[1:])


def test_status_panel_shows_observations_without_simulated_vehicle_or_commands(view):
    displayed=[]
    original=view.text_tile
    view.text_tile=lambda value,*args: (displayed.append(str(value)),original(value,*args))[1]
    snapshot=make_snapshot(main=[{'class_name':'car','bbox':(10,20,80,100),'confidence':.9}],
                           signal=SignalObservation(light=Light.UNKNOWN,reason='observed'))
    render(view,snapshot=snapshot,navigation={'valid':False,'direction':'none',
                                               'reason':'unconfirmed'})
    joined='\n'.join(displayed)
    assert '感知状态' in joined
    assert '道路目标' in joined and '1 个' in joined
    assert '信号灯 未确认' in joined
    assert '模拟车辆' not in joined
    assert '行驶中' not in joined
    assert '已停止' not in joined
    assert '请求左转' not in joined
    assert '车速' not in joined


def test_binding_arguments_do_not_create_manual_selection_ui(view):
    with_binding=render(view,binding=True,drag_box=(10,10,100,100))
    assert view.hit(1247,685) is None
    assert view.hit(1000,600) is None
    without_binding=render(view,binding=False)
    assert np.array_equal(with_binding,without_binding)


def test_read_only_render_does_not_require_simulation_controls(view):
    image=view.render(np.zeros((360,640,3),np.uint8),make_snapshot(),frame_id=0,fps=0)
    assert image.shape==(900,1600,3)
    assert view.hit(1214,41)=='o'


def test_live_source_timestamp_drives_navigation_age_and_timeline_with_zero_fps(view):
    displayed=[]
    original=view.text_tile
    view.text_tile=lambda value,*args: (displayed.append(str(value)),original(value,*args))[1]
    snapshot=make_snapshot()
    view.render(np.zeros((360,640,3),np.uint8),snapshot,
                frame_id=42,fps=0,source_timestamp_ns=6_000_000_000,
                source_start_timestamp_ns=5_000_000_000,
                navigation={'valid':True,'direction':'left','distance_m':12,
                            'road_name':'测试路','source_frame_id':42,
                            'timestamp_ns':5_900_000_000})
    assert '00:01' in displayed
    assert 'LIVE' in displayed
    assert any('12 米后左转' in value for value in displayed)


def test_live_navigation_without_a_valid_source_clock_is_not_trusted(view):
    displayed=[]
    original=view.text_tile
    view.text_tile=lambda value,*args: (displayed.append(str(value)),original(value,*args))[1]
    view.render(np.zeros((360,640,3),np.uint8),make_snapshot(),frame_id=42,fps=0,
                navigation={'valid':True,'direction':'left','distance_m':12,
                            'source_frame_id':42,'timestamp_ns':0})
    assert any('导航暂不可读' in value for value in displayed)


def test_live_view_hides_source_and_playback_controls_but_keeps_display_actions(view):
    render(view,live=True)
    assert view.hit(1214,41) is None
    assert view.hit(1346,41) == 'f'
    assert view.hit(1497,41) == 'c'
    assert [view.hit(x,685) for x in (537,1026,1117)] == [None, None, None]
    assert view.hit(1437,685) == 'v'


@pytest.mark.parametrize(('traffic','evidence','expected'), [
    ('RED', 'ONE_DETECTED_RED_SIGNAL', '红灯'),
    ('GREEN', 'ONE_DETECTED_GREEN_SIGNAL', '绿灯'),
    ('OFF', 'YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL', '黄灯'),
    ('OFF', 'NO_SIGNAL_ROI', '未确认'),
])
def test_fpga_perception_directly_controls_displayed_light_and_does_not_bind_a_lane(
        view, traffic, evidence, expected):
    displayed=[]
    original=view.text_tile
    view.text_tile=lambda value,*args: (displayed.append(str(value)),original(value,*args))[1]
    snapshot=make_snapshot(signal=SignalObservation(light=Light.RED, associated=False,
                                                     reason='unbound'))
    render(view,snapshot=snapshot,fpga_perception={
        'traffic': traffic, 'traffic_evidence': evidence,
        'lane_left': 'NONE', 'lane_right': 'NONE',
    })
    joined='\n'.join(displayed)
    assert f'信号灯 {expected}' in joined
    assert '车道关联' in joined and '未关联' in joined
    assert '控制灯' not in joined


def test_diagnostics_uses_transport_status_instead_of_fixed_link_claim(view):
    displayed=[]
    original=view.text_tile
    view.text_tile=lambda value,*args: (displayed.append(str(value)),original(value,*args))[1]
    render(view,diagnostics=True,transport_status='BAR1 applied seq=4')
    joined='\n'.join(displayed)
    assert 'BAR1 applied seq=4' in joined
    assert 'PCIe / FPGA 未联机' not in joined


def test_live_diagnostic_buttons_require_enable_and_are_hidden_outside_diagnostics(view):
    status={'enabled':False,'phase':'DISABLED','sequence':0,'reason':'explicit_enable_and_click_required'}
    render(view,live=True,diagnostics=True,test_status=status)
    assert view.hit(880,560)=='fpga_test_enable'
    assert [view.hit(x,560) for x in (1050,1215,1375)]==[None,None,None]
    status.update(enabled=True,phase='IDLE')
    render(view,live=True,diagnostics=True,test_status=status)
    assert [view.hit(x,560) for x in (1050,1215,1375)]==[
        'fpga_test_left','fpga_test_right','fpga_test_off']
    render(view,live=True,diagnostics=False,test_status=status)
    assert not any(key.startswith('fpga_test_') for key,_ in view.buttons)
