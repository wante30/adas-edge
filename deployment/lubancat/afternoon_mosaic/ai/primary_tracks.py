"""Shared measured identities. Per-camera association; predictions stay internal."""
from tools.competition_console.byte_tracker_lite import ByteTracker
from forward_view import source_id

class PrimaryTracks:
    def __init__(self):
        self.context=None
        self.trackers={}
        self.current=[]

    def update(self,detections,context,timestamp_ns):
        if context!=self.context:
            self.context=context
            self.trackers={q:ByteTracker(track_buffer=5,high_thresh=.35,new_track_thresh=.35) for q in ("Q1","Q2","Q3","Q4")}
        groups={q:[] for q in self.trackers}
        output=[]
        for original in detections[:32]:
            b=original.get("bbox")
            if not b or len(b)!=4: continue
            x1,y1,x2,y2=map(float,b)
            if context[0]=="ROAD_Q1": q=source_id(context)
            else:
                q="Q%d"%(1+(int((x1+x2)/2>=640))+(2*int((y1+y2)/2>=360)))
                # A seam-straddling box has no safe camera identity.
                ox=640 if q in ("Q2","Q4") else 0;oy=360 if q in ("Q3","Q4") else 0
                if not (ox<=x1<x2<=ox+640 and oy<=y1<y2<=oy+360): q="UNASSIGNED"
            det=dict(original,source_id=q,track_id=0,measurement_ns=timestamp_ns)
            if q in groups:
                row=dict(x=x1,y=y1,width=x2-x1,height=y2-y1,className=det.get("class_name",""),confidence=det.get("confidence",0),original=det)
                groups[q].append(row)
            output.append(det)
        for q,rows in groups.items():
            for row in self.trackers[q].update(rows,timestamp_ns):
                row["original"]["track_id"]=row["trackId"]
                vx,vy,vw,vh=row.get("velocity",[0,0,0,0])
                row["original"]["velocity"]=[vx-vw/2,vy-vh/2,vw,vh]
        self.current=output
        return output
