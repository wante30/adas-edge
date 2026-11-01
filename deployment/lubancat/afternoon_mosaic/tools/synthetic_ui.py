"""Explicit synthetic display/control fixture; AI protocol only, no control ports."""
import json,os,socket,struct,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"ai"))
os.environ["ADAS_CONTROL_MODE"]="DRY_RUN"
from ai_bridge import encode_ai_result_packet,build_dynamic_status
from risk_events import EventStore,RiskObservation
from auto_brake import AutoBrakeController
from control_scenarios import crosswalk_pulse_candidate

def main():
    transport=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);transport.bind(("127.0.0.1",38260));transport.settimeout(.5)
    bridge_id="SYNTHETIC-"+uuid.uuid4().hex;mode="ROAD_Q1";epoch=0
    store=EventStore();control=AutoBrakeController(enabled=True,journal=store.journal)
    start=None;last_ns=0
    while start is None or time.monotonic()-start<24:
        try:packet,peer=transport.recvfrom(40960)
        except socket.timeout:continue
        def identity(fid=None):
            r=dict(mode=mode,epoch=str(epoch),bridge_id=bridge_id,box_space="ROAD_VIEW_1280x720" if mode=="ROAD_Q1" else "MOSAIC_1280x720")
            if fid is not None:r["frame_id"]=fid
            return r
        if packet[:4]==b"MVI1":
            _,epoch,index=struct.unpack("<4sQI",packet);mode="ROAD_Q1" if index==0 else "MOSAIC_FULL"
            transport.sendto(b"MIA1"+json.dumps(dict(identity(),state="APPLIED",request_epoch=str(epoch))).encode(),peer);continue
        if packet[:4]!=b"MVR3" or len(packet)!=36:continue
        _,fid,ts,request_epoch,_,_=struct.unpack("<4sQQQff",packet)
        if request_epoch!=epoch:continue
        now=time.monotonic_ns()
        if start is None:start=time.monotonic()
        elapsed=time.monotonic()-start
        if now-last_ns<80000000:time.sleep((80000000-(now-last_ns))/1e9)
        last_ns=time.monotonic_ns();now=last_ns
        if elapsed<5:stage="warning";kind="FRONT_COLLISION_RISK";cls="car";severity="warning"
        elif elapsed<10:stage="critical";kind="FRONT_COLLISION_RISK";cls="car";severity="critical"
        elif elapsed<15:stage="pedestrian";kind="PEDESTRIAN_CROSSING_PATH";cls="person";severity="critical"
        elif elapsed<19:stage="pulse";kind="CROSSWALK_BRAKE_PULSE";cls="person";severity="critical"
        else:stage="cleared";kind="";cls="person";severity="warning"
        # Synthetic event/bbox overlays are labelled on screen and in every row.
        bbox=[260,170,380,330]
        candidates=[dict(kind=kind,severity=severity,track_id=7,source_id="Q1",target_class=cls,bbox=bbox,confidence=.99,path_valid=True,synthetic=True,evidence="SYNTHETIC_UI_CONTROL_FIXTURE")] if kind else []
        events=store.update(candidates,mode=mode,epoch=epoch,bridge_id=bridge_id,model_id="SYNTHETIC-DRY-RUN",frame_id=fid,timestamp_ns=ts,now_ns=now)
        control.update(RiskObservation(events=events),context=identity(),now_ns=now)
        status=build_dynamic_status(source="SYNTHETIC_UI_TEST",frame_id=fid,timestamp_ns=ts,now_ns=now,hud=dict(traffic="OFF"),navigation={},hazard_timestamp_ns=ts,hazards=events,detections=[],signal_age_ms=None,main_age_ms=0,specialist_age_ms=None,canonical_risks=True)
        status.update(inference=identity(fid),layout="quad",display_view="front" if mode=="ROAD_Q1" else "quad",road_quadrant=0,
            model_selection=dict(active_id="SYNTHETIC-DRY-RUN",active_label="SYNTHETIC / 合成事件验证",state="READY",message="合成目标 / 无 NPU 验收数据"),
            control=control.last_result or dict(control_mode="DRY_RUN"),box_identities=[dict(track_id=7,source_id="Q1",velocity=[0,0,0,0])],synthetic=True,synthetic_stage=stage)
        factor=2 if mode=="ROAD_Q1" else 1
        x1,y1,x2,y2=[v*factor for v in bbox]
        boxes=[(x1,y1,x2,y2,.3,.8,.9,0 if cls=="person" else 2,.99)]
        payload=encode_ai_result_packet(frame_id=fid,timestamp_ns=ts,flags=0,nav_direction=0,nav_distance_m=-1,boxes=boxes,status=status)
        transport.sendto(payload,peer)
        store.journal.write("synthetic_ui_stage",synthetic=True,frame_id=fid,stage_name=stage,events=list(events),control=status["control"])
    control.close();store.journal.close();transport.close()
if __name__=="__main__":main()
