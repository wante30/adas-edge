"""Bounded numerical rules over measured tracks; no image inference or optical flow."""
from collections import deque
import math
import os
import time

VEHICLES={"car","truck","bus"}
TWO_WHEELERS={"motorcycle","bicycle"}
RULES=("front","pedestrian","two_wheeler","cut_in","oncoming","braking")

class LightRiskEngine:
    def __init__(self):
        self.context=None;self.history={};self.confirm={}
        self.enabled={r:os.environ.get("ADAS_RULE_"+r.upper(),"1")!="0" for r in RULES}
        self.stats={r:dict(count=0,events=0,samples=deque(maxlen=2048)) for r in RULES}

    @staticmethod
    def in_path(x,foot,margin=0):
        half=.03+.23*max(0.,min(1.,(foot-.48)/.5))
        return foot>=.48 and abs(x-.5)<=half+margin

    def _run(self,rule,condition,kind,critical,d,evidence,events):
        if not self.enabled[rule]:return
        started=time.perf_counter_ns()
        key=(d["source_id"],d["track_id"],kind)
        self.confirm[key]=(self.confirm.get(key,0)+1 if condition else 0)
        if condition and self.confirm[key]>=2:
            events.append(dict(kind=kind,severity="critical" if critical and self.confirm[key]>=3 else "warning",
                source_id=d["source_id"],track_id=d["track_id"],target_class=d["class_name"],
                bbox=d["bbox"],confidence=d["confidence"],path_valid=True,evidence=evidence,
                synthetic=bool(d.get("synthetic",False))))
            self.stats[rule]["events"]+=1
        st=self.stats[rule];st["count"]+=1;st["samples"].append((time.perf_counter_ns()-started)/1e6)

    def update(self,detections,timestamp_ns,context,semantics=None):
        if context!=self.context:
            self.context=context;self.history={};self.confirm={}
        events=[];seen=set();semantics=semantics or {}
        for original in detections[:32]:
            if original.get("source_id")!="Q1" or not original.get("track_id"):continue
            if original.get("predicted") or original.get("confidence",0)<.4:continue
            d=dict(original);box=list(map(float,d["bbox"]))
            if not all(math.isfinite(v) for v in box):continue
            # Q1 mosaic and road inference both measure the same native 640x360 image.
            if not (0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=360):continue
            d["bbox"]=box
            x=(box[0]+box[2])/1280;foot=box[3]/360;h=(box[3]-box[1])/360
            w=(box[2]-box[0])/640;area=h*w;cls=d["class_name"]
            key=("Q1",d["track_id"]);seen.add(key)
            history=self.history.setdefault(key,deque(maxlen=16))
            if history and timestamp_ns<=history[-1][0]:continue
            if history and timestamp_ns-history[-1][0]>400000000:history.clear()
            history.append((timestamp_ns,x,foot,h))
            reference=next((r for r in history if .25<=(timestamp_ns-r[0])/1e9<=.8),None)
            dt=(timestamp_ns-reference[0])/1e9 if reference else 0
            vx=(x-reference[1])/dt if dt else 0
            growth=(h-reference[3])/max(.03,reference[3])/dt if dt else 0
            in_path=self.in_path(x,foot);near_path=self.in_path(x,foot,.12)
            sustained=bool(reference and len(history)>=3)
            closing=sustained and growth>.35
            near=foot>.72 and h>.20 and area>.012
            vehicle=cls in VEHICLES and w/max(h,.01)<2.7
            towards=sustained and (x-.5)*vx<0 and abs(vx)>.08
            crossing=towards and abs(vx)>.12
            previous_outside=bool(reference and not self.in_path(reference[1],reference[2]))
            self._run("front",vehicle and in_path and near,"FRONT_COLLISION_RISK",
                closing and foot>.84 and h>.28,d,"MEASURED_BOX_GROWTH_ASSUMED_PATH" if closing else "MEASURED_NEAR_ASSUMED_PATH",events)
            person=cls=="person"
            self._run("pedestrian",person and near_path and foot>.56,"PEDESTRIAN_IN_PATH",
                in_path and sustained and foot>.80 and h>.23,d,"MEASURED_PERSON_ASSUMED_PATH",events)
            self._run("pedestrian",person and crossing and near_path,"PEDESTRIAN_CROSSING_PATH",
                in_path and foot>.75 and h>.20,d,"MEASURED_LATERAL_ENTRY_ASSUMED_PATH",events)
            self._run("two_wheeler",cls in TWO_WHEELERS and near_path and foot>.58,"TWO_WHEELER_PATH_CONFLICT",
                sustained and in_path and foot>.78 and h>.22 and (towards or closing or foot>.88),d,"MEASURED_TWO_WHEELER_PATH_CONFLICT",events)
            self._run("cut_in",vehicle and towards and near_path and previous_outside,"VEHICLE_CUT_IN",
                in_path and near and foot>.80,d,"MEASURED_SIDE_TO_PATH_ENTRY",events)
            direction=d.get("direction")=="oncoming"
            self._run("oncoming",vehicle and direction and near_path,"ONCOMING_PATH_CONFLICT",
                bool(d.get("direction_verified")) and in_path and near and closing,d,
                "INDEPENDENT_DIRECTION_CONFIRMED" if d.get("direction_verified") else "DIRECTION_UNVERIFIED_NO_BRAKE",events)
            lamp=semantics.get("lamps",{}).get(str(d["track_id"]),{})
            lamp_fresh=0<=timestamp_ns-lamp.get("timestamp_ns",-10**18)<=1000000000
            self._run("braking",vehicle and in_path and bool(lamp.get("rise")) and lamp_fresh,
                "SUSPECTED_BRAKING_WITH_CLOSING",closing and near and foot>.84,d,"PAIRED_RED_LAMP_RISE_AND_MEASURED_CLOSING",events)
        self.history={k:v for k,v in self.history.items() if timestamp_ns-v[-1][0]<800000000}
        self.confirm={k:v for k,v in self.confirm.items() if k[:2] in seen}
        return events

    def metrics(self):
        out={}
        for rule,st in self.stats.items():
            a=sorted(st["samples"])
            out[rule]=dict(enabled=self.enabled[rule],count=st["count"],events=st["events"],
                mean_ms=sum(a)/len(a) if a else 0,p95_ms=a[int(.95*(len(a)-1))] if a else 0)
        return out
