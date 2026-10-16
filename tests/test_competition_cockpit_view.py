"""Behavior checks for clickable native controls and image-space selection."""
from dataclasses import replace
import numpy as np
import pytest
from competition_ai.bound_signal import SignalObservation
from competition_ai.scenario import Controls, Mode, Perception
from competition_ai.scenario_policy import decide
from tools.competition_console.pipeline import FrontSnapshot
from tools.competition_console.view import NativeView


@pytest.fixture
def view():
    return NativeView()


def render(view, mode=Mode.INTERSECTION, shape=(1080,1920,3), **kwargs):
    controls=Controls(mode=mode)
    snapshot=FrontSnapshot([],[],SignalObservation(),None,None,'UNBOUND')
    return view.render(np.zeros(shape,np.uint8),snapshot,controls,decide(controls,Perception(),0),
                       frame_id=0,fps=30,**kwargs)


@pytest.mark.parametrize('mode',list(Mode))
def test_all_top_tabs_remain_clickable_including_duplicate_bottom_actions(view,mode):
    render(view,mode)
    assert [view.hit(x,41) for x in (538,703,868,1033)] == ['i','d','n','h']
    assert view.hit(201,842) is not None
    assert view.hit(1500,70) is None


def test_four_by_three_source_uses_letterbox_and_selection_matches_pixels(view):
    image=render(view,shape=(480,640,3))
    assert image.shape==(900,1600,3)
    x,y,w,h=view.content_rect
    assert abs(w/h-640/480)<.003
    assert view.video_point(x-1,y+100,(480,640,3)) is None
    px,py=view.video_point(x+w/2,y+h/2,(480,640,3))
    assert px==pytest.approx(320) and py==pytest.approx(240)
    assert view.video_point(600,680,(480,640,3)) is None


def test_diagnostics_keep_mode_controls_and_player_actions_accessible(view):
    render(view,diagnostics=True)
    assert view.hit(868,41)=='n'
    assert view.hit(1247,685)=='b'


def test_binding_returns_from_other_views_without_changing_driving_controls():
    from tools.competition_console.controls import Controller
    c=Controller()
    old=c.controls
    c.key('c');c.key('f');c.key('b')
    assert c.binding and not c.bev and not c.diagnostics and c.controls==old
