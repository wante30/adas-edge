"""Navigation observations appear independently of driving permission."""
import numpy as np
import pytest

from competition_ai.bound_signal import SignalObservation
from competition_ai.scenario import Controls, Direction, Mode, Perception
from competition_ai.scenario_policy import decide
from tools.competition_console.pipeline import FrontSnapshot
from tools.competition_console.view import NativeView


@pytest.fixture
def display(monkeypatch):
    view=NativeView()
    seen=[]
    original=view.text_tile
    def capture(value,*args):
        seen.append(value)
        return original(value,*args)
    monkeypatch.setattr(view,'text_tile',capture)
    return view,seen


def render(view,observation=None,mode=Mode.NAVIGATION,**kwargs):
    controls=Controls(mode=mode,signalized=True,navigation=Direction.RIGHT,navigation_valid=True)
    p=Perception()
    s=FrontSnapshot([],[],SignalObservation(),None,None,'UNBOUND')
    image=np.full((540,960,3),(25,60,100),np.uint8)
    return view.render(image,s,controls,decide(controls,p,1_000_000_000),frame_id=30,fps=30,
                       navigation=observation,navigation_image=np.full((540,960,3),(90,70,20),np.uint8),**kwargs)


def nav(**changes):
    result={'direction':'right','distance_m':893,'road_name':'白沙大道','valid':True,
            'confidence':.98,'reason':'observed','source_frame_id':27,'timestamp_ns':900_000_000,
            'raw_text':'893 米 白沙大道','inference_ms':24.2}
    result.update(changes)
    return result


def test_recognized_navigation_is_visible_while_unassociated_light_still_stops(display):
    view,text=display
    image=render(view,nav())
    assert image.shape==(900,1600,3)
    assert '893 米后右转' in text and '白沙大道' in text
    assert '同帧导航识别' in text and '行驶建议 · 灯控待确认' in text
    assert '行驶状态' in text and '未测量' in text and '仅导航提示' in text
    assert '模拟车辆' not in text and '已停止' not in text and 'P' not in text
    assert '识别源帧 27 · 100 ms' in text
    assert view.hit(868,41)=='n'
    assert view.video_point(230,510,(540,960,3)) is None


def test_navigation_panel_does_not_occlude_the_road_evidence(display):
    view,_=display
    baseline=render(view)
    actual=render(view,nav())
    assert np.array_equal(actual[106:727,464:1568],baseline[106:727,464:1568])


@pytest.mark.parametrize('observation', [
    nav(valid=False,reason='side_view'), nav(direction='none'),
    nav(source_frame_id=31), nav(timestamp_ns=1_000_000_001),
    nav(source_frame_id=None),
])
def test_unreadable_or_future_navigation_does_not_reuse_old_turn_distance_or_road(display,observation):
    view,text=display
    render(view,nav())
    text.clear()
    render(view,observation)
    assert '导航暂不可读' in text
    assert '893 米后右转' not in text and '白沙大道' not in text


def test_uturn_is_preserved_as_route_hint_without_becoming_a_left_turn(display):
    view,text=display
    render(view,nav(direction='uturn',distance_m=221,road_name=''))
    assert '221 米后掉头' in text
    assert '行驶建议 · 灯控待确认' in text


def test_navigation_diagnostics_report_the_connected_source_and_age(display):
    view,text=display
    render(view,nav(),diagnostics=True)
    assert '导航解析：同帧导航屏识别 / 有效' in text
    assert any('导航源帧：27 / 龄期：100 ms' in value for value in text)
    assert not any('导航解析：未接入' in value for value in text)


def test_ended_navigation_has_no_live_instruction(display):
    view,text=display
    render(view,nav(),ended=True)
    assert '导航已停止更新' in text
    assert '893 米后右转' not in text


def test_other_modes_keep_their_existing_presentation(display):
    view,text=display
    baseline=render(view,mode=Mode.INTERSECTION)
    actual=render(view,nav(),mode=Mode.INTERSECTION)
    assert np.array_equal(actual,baseline)
    assert '模拟车辆' in text and '已停止' in text and 'P' in text
