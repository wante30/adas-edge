"""Latest-slot <=2 Hz reuse of existing MTL masks and bounded target crops."""
from collections import deque
import threading,time
from forward_view import source_id, local_detection
from competition_ai.road_geometry import crosswalk_polygons
from competition_ai.hazard_events import paired_red_lamps

class SemanticWorker:
    def __init__(self):
        self.cv=threading.Condition();self.pending=None;self.stopped=False
        self.next_ns=0;self.result={};self.lamps={};self.context=None
        self.samples=deque(maxlen=2048);self.count=0;self.errors=0
        self.thread=threading.Thread(target=self._run,name="light-semantics",daemon=True);self.thread.start()

    def submit(self,frame,detections,timestamp_ns,context,masks=None,letterbox=None):
        with self.cv:
            if context!=self.context:
                self.context=context;self.pending=None;self.result={};self.next_ns=0
            if timestamp_ns<self.next_ns:return
            self.next_ns=timestamp_ns+500000000
            # CapiMtlDetector allocates fresh masks for each completion; frame is immutable.
            self.pending=(frame,[local_detection(d,context) for d in detections if d.get("source_id")==source_id(context)][:32],timestamp_ns,context,masks,letterbox)
            self.cv.notify()

    def _process(self,frame,detections,ts,context,masks,letterbox):
        if getattr(self,"processed_context",None)!=context:
            self.processed_context=context;self.lamps={}
        if letterbox is not None and not isinstance(letterbox,dict):
            letterbox=dict(zip(("scale","pad_x","pad_y"),letterbox))
        crosswalks=[]
        if context[0]=="ROAD_Q1" and masks is not None and letterbox is not None:
            crosswalks=[c for c in crosswalk_polygons(masks[0],masks[0].shape,letterbox,640,360)
                if c["confidence"]>=.45 and c["bbox"][2]-c["bbox"][0]>=.16 and c["center_y"]>.35]
        lamps={}
        targets=sorted((d for d in detections if d.get("class_name") in ("car","truck","bus") and d.get("track_id")),key=lambda d:d["bbox"][3],reverse=True)[:3]
        for d in targets:
            value=paired_red_lamps(frame,d["bbox"]);key=str(d["track_id"])
            previous=self.lamps.get(key)
            rise=bool(previous is not None and value>.035 and value-previous>.02)
            lamps[key]=dict(value=value,rise=rise,timestamp_ns=ts);self.lamps[key]=value
        return dict(context=context,timestamp_ns=ts,crosswalks=crosswalks,lamps=lamps,mtl_available=masks is not None and context[0]=="ROAD_Q1")

    def _run(self):
        while True:
            with self.cv:
                while self.pending is None and not self.stopped:self.cv.wait()
                if self.stopped:return
                job=self.pending;self.pending=None
            started=time.perf_counter_ns()
            try: result=self._process(*job)
            except Exception as exc:
                self.errors+=1;result=dict(context=job[3],timestamp_ns=job[2],crosswalks=[],lamps={},error=type(exc).__name__)
            with self.cv:
                if job[3]==self.context:self.result=result
            self.samples.append((time.perf_counter_ns()-started)/1e6);self.count+=1

    def current(self,ts,context):
        with self.cv:
            result=self.result
            if result.get("context")!=context or not 0<=ts-result.get("timestamp_ns",-10**18)<=900000000:return {}
            return result

    def metrics(self):
        now=time.monotonic_ns()
        if now-getattr(self,"metrics_ns",-1000000000)<1000000000:return self.metrics_cache
        self.metrics_ns=now
        a=sorted(self.samples)
        self.metrics_cache=dict(count=self.count,errors=self.errors,max_hz=2,mean_ms=sum(a)/len(a) if a else 0,p95_ms=a[int(.95*(len(a)-1))] if a else 0)
        return self.metrics_cache

    def close(self):
        with self.cv:self.stopped=True;self.cv.notify()
        self.thread.join(timeout=2)
