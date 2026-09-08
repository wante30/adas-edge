import numpy as np
import pytest
from competition_ai import CompetitionAdas, SceneInput, TrafficLightState, classify_traffic_light
from competition_io import DetectionRecord, FrameDescriptor, ResultDescriptor, MockTransport, MockRoundRobin
from competition_io.protocol import ProtocolError

def test_traffic_light_hsv_red_green_yellow_and_unknown():
    for color, expected in [((0,0,255),TrafficLightState.RED),((0,255,0),TrafficLightState.GREEN),((0,255,255),TrafficLightState.YELLOW)]:
        frame=np.zeros((20,20,3),np.uint8); frame[5:15,5:15]=color
        assert classify_traffic_light(frame, {'bbox':[5,5,15,15],'confidence':.9}) == expected
    assert classify_traffic_light(np.zeros((20,20,3),np.uint8), {'bbox':[5,5,15,15],'confidence':.9}) == TrafficLightState.UNKNOWN
    assert classify_traffic_light(np.zeros((20,20,3),np.uint8), {'bbox':[-4,-4,1,1],'confidence':.9}) == TrafficLightState.UNKNOWN

def test_scene_state_machine_rules():
    adas=CompetitionAdas()
    assert adas.decide(SceneInput('red', vehicle_command='go')).action == 'STOP'
    assert 'RED_LIGHT' in adas.decide(SceneInput('red', vehicle_command='go')).warning_flags
    assert adas.decide(SceneInput('green', vehicle_command='go')).action == 'GO'
    assert adas.decide(SceneInput('unknown')).action == 'STOP'
    assert adas.decide(SceneInput('yellow', vehicle_command='go')).action == 'STOP'
    assert adas.decide(SceneInput('red', vehicle_command='stop')).warning_flags == ()
    assert adas.decide(SceneInput('green','dashed','left','change_left')).action == 'LEFT'
    assert 'SOLID_LANE_CHANGE_FORBIDDEN' in adas.decide(SceneInput('green','solid','right','change_right')).warning_flags
    assert adas.decide(SceneInput('green',pedestrian_hazard=True)).action == 'BRAKE'
    assert adas.decide(SceneInput('green',obstacle_hazard=True)).action == 'BRAKE'

def test_protocol_round_trip_and_stream_routing():
    f=FrameDescriptor(2,8,99,1280,720,1,2560,b'abc'); assert FrameDescriptor.deserialize(f.serialize()) == f
    r=ResultDescriptor(2,8,99,(DetectionRecord(2,.75,1,2,3,4),),1,2,3,4); rr=ResultDescriptor.deserialize(r.serialize()); assert (rr.stream_id,rr.frame_id,rr.timestamp_ns,rr.warning_flags)==(2,8,99,4); assert abs(rr.detections[0].confidence-.75)<1e-4
    t=MockTransport(); t.send_frame(f); assert t.recv_frame(2)==f; t.send_result(r); got=t.recv_result(2); assert got.stream_id==r.stream_id and got.frame_id==r.frame_id and abs(got.detections[0].confidence-r.detections[0].confidence)<1e-4; assert t.recv_frame(3) is None

@pytest.mark.parametrize('bad', [b'XXXX'+b'\0'*50, b'ADSF'+b'\0'*4])
def test_protocol_rejects_bad_packets(bad):
    with pytest.raises(ProtocolError): FrameDescriptor.deserialize(bad)

def test_four_stream_round_robin_preserves_identity_and_drops_oldest():
    s=MockRoundRobin(4,2)
    for sid in range(4):
        for fid in range(3): s.push(FrameDescriptor(sid,fid,0,1,1,1,2,b'x'))
    seen=[]
    while (f:=s.pop()) is not None: seen.append((f.stream_id,f.frame_id))
    assert {sid for sid,_ in seen}=={0,1,2,3}; assert s.dropped==[1,1,1,1]
    assert [sid for sid,_ in seen[:4]]==[0,1,2,3]
