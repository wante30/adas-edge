"""One nonblocking decision chain for fresh canonical forward critical events."""
from __future__ import annotations
import json,os,socket,time,uuid
from pathlib import Path
from risk_events import BRAKE_TYPES,EventJournal

BRAKE_KINDS=BRAKE_TYPES
SEVERITIES={"critical"}

class AutoBrakeController:
    def __init__(self,*,enabled=None,port=38262,journal=None,forward_source_id="Q1"):
        self.enabled=(os.environ.get("ADAS_AUTO_BRAKE")=="1") if enabled is None else bool(enabled)
        self.mode=os.environ.get("ADAS_CONTROL_MODE","HARDWARE")
        self.authorized=os.environ.get("ADAS_CONTROL_AUTHORIZED","1" if self.enabled else "0")=="1"
        self.forward_source_id=forward_source_id
        self.address=("127.0.0.1",int(port))
        if self.mode=="DRY_RUN":
            from control_transport import DryRunSocket
            self.sock=DryRunSocket()
        else:self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.sock.settimeout(0.)
        self.journal=journal or EventJournal();self.pending=None
        self.last_result=None;self.last_send={};self.pulses={};self.previous=set();self.pulse_sent=set()
        self.vehicle_cache={};self.vehicle_read_ns=0

    def vehicle_state(self,now_ns):
        if self.mode=="DRY_RUN":
            return dict(available=True,authorized=True,simulated=True,source="DRY_RUN_VIRTUAL_VEHICLE",timestamp_ns=now_ns)
        if now_ns-self.vehicle_read_ns<100000000:return self.vehicle_cache
        self.vehicle_read_ns=now_ns
        try:
            path=Path(os.environ.get("LVT1_STATUS_FILE","/dev/shm/adas_vehicle_status.json"))
            age=time.time()-path.stat().st_mtime
            data=json.loads(path.read_text(encoding="utf-8")) if 0<=age<=.4 and path.stat().st_size<=8192 else {}
            available=(data.get("schema")=="pango.rk3566.vehicle_status.v1" and data.get("online") is True
                and not data.get("sensor_stale",True) and not data.get("simulated",False))
            self.vehicle_cache=dict(available=available,authorized=available and not data.get("brake_deny",False),simulated=False,source="LVT1",timestamp_ns=now_ns)
        except (OSError,ValueError):self.vehicle_cache=dict(available=False,authorized=False,source="LVT1_UNAVAILABLE")
        return self.vehicle_cache

    def _reason(self,e,now,context,vehicle):
        if e.get("lifecycle_state")!="ACTIVE" or not e.get("valid",False):return "event_cleared_or_invalid"
        if not self.enabled or not self.authorized:return "control_not_authorized"
        if self.mode not in ("DRY_RUN","HARDWARE"):return "invalid_control_mode"
        if e.get("source_id")!=self.forward_source_id or e.get("track_id",0)<=0:return "not_measured_forward_target"
        if e.get("kind") not in BRAKE_TYPES:return "unsupported_event_type"
        if not 0<=now-e.get("timestamp_ns",0)<=400000000 or e.get("expires_at_ns",0)<=now:return "stale_event"
        if e.get("predicted") or not e.get("control_eligible") or not e.get("path_valid"):return "ineligible_or_invalid_path"
        if context and (e.get("inference_mode")!=context.get("mode") or str(e.get("mode_epoch"))!=str(context.get("epoch"))):return "context_mismatch"
        if not vehicle.get("available"):return "vehicle_state_unavailable"
        if not vehicle.get("authorized"):return "vehicle_control_denied"
        if e.get("kind")=="CROSSWALK_BRAKE_PULSE" and (self.mode!="DRY_RUN" or not e.get("synthetic")):return "pulse_test_only"
        return None

    def _record(self,stage,event,now,**fields):
        row=dict(event_id=event.get("event_id"),track_id=event.get("track_id"),timestamp_ns=now,control_mode=self.mode,
            event_type=event.get("kind"),severity=event.get("severity"),source_id=event.get("source_id"),
            lifecycle_state=event.get("lifecycle_state"),frame_id=event.get("frame_id"),
            event_timestamp_ns=event.get("timestamp_ns"),mode_epoch=event.get("mode_epoch"),**fields)
        self.journal.write(stage,**row)
        return row

    def _poll(self,now):
        if not self.pending:return
        request=self.pending
        try:
            payload,_=self.sock.recvfrom(4096);reply=json.loads(payload.decode("utf-8"))
            if reply.get("request_id")!=request["request_id"]:
                self.journal.write("brake_ack_unmatched",timestamp_ns=now,reply=reply);return
            ok=bool(reply.get("ok") and reply.get("accepted",reply.get("ok")))
            self.last_result=dict(reply,stage="brake_accepted" if ok else "brake_rejected",event_ids=[e["event_id"] for e in request["events"]],executed=bool(reply.get("executed",False)))
            for e in request["events"]:
                self._record(self.last_result["stage"],e,now,request_id=request["request_id"],reason="dry_run_ack" if self.mode=="DRY_RUN" else reply.get("reason","hardware_ack"),executed=self.last_result["executed"],simulated=self.mode=="DRY_RUN",
                    fpga_accepted=reply.get("fpga_accepted"),fpga_action_active=reply.get("fpga_action_active"),
                    hardware_sequence=reply.get("sequence"),vehicle_feedback=reply.get("vehicle_feedback"))
                if ok and e["kind"]=="CROSSWALK_BRAKE_PULSE":
                    self.pulses[e["event_id"]]=(now+300000000,e,request["request_id"])
                    self._record("pulse_simulated_execution",e,now,request_id=request["request_id"],simulated=True,executed=False,duration_ms=300)
            self.pending=None
        except BlockingIOError:pass
        except (OSError,ValueError) as exc:
            if not isinstance(exc,(BlockingIOError,socket.timeout)):
                self.last_result=dict(stage="brake_rejected",reason=type(exc).__name__,control_mode=self.mode)
                for e in request["events"]:self._record("brake_rejected",e,now,request_id=request["request_id"],reason=type(exc).__name__)
                self.pending=None
        if self.pending and now-request["sent_ns"]>250000000:
            for e in request["events"]:self._record("brake_timeout",e,now,request_id=request["request_id"],reason="ack_timeout")
            self.last_result=dict(stage="brake_timeout",request_id=request["request_id"],control_mode=self.mode,executed=False);self.pending=None

    def update(self,hazards,context=None,vehicle=None,now_ns=None):
        now=time.monotonic_ns() if now_ns is None else now_ns
        self._poll(now)
        events=getattr(hazards,"events",()) or ();active_ids={e["event_id"] for e in events if e.get("lifecycle_state")=="ACTIVE"}
        for event_id,(end,e,request_id) in list(self.pulses.items()):
            if now>=end or event_id not in active_ids:
                self._record("pulse_ended",e,now,request_id=request_id,reason="duration_elapsed" if now>=end else "event_cleared",executed=False,simulated=True)
                self.last_result=dict(stage="pulse_ended",event_id=event_id,request_id=request_id,control_mode=self.mode,executed=False)
                del self.pulses[event_id]
        for event_id in self.previous-active_ids:
            self.journal.write("control_event_cleared",event_id=event_id,timestamp_ns=now,control_mode=self.mode)
        if self.previous and not active_ids and (self.last_result or {}).get("stage") in ("brake_accepted","brake_rejected"):
            self.last_result=dict(stage="control_event_cleared",control_mode=self.mode,executed=False)
        self.pulse_sent.intersection_update(active_ids)
        self.previous=active_ids
        vehicle=self.vehicle_state(now) if vehicle is None else vehicle
        candidates=[]
        for e in events:
            if e.get("severity")!="critical":continue
            reason="pulse_already_requested" if e["event_id"] in self.pulse_sent else self._reason(e,now,context,vehicle)
            if not reason and now-self.last_send.get(e["event_id"],-10**18)<400000000:reason="request_cooldown"
            if not reason and self.pending:reason="awaiting_ack"
            self._record("control_decision",e,now,reason=reason or "eligible",eligible=reason is None,vehicle_source=vehicle.get("source"),vehicle_simulated=bool(vehicle.get("simulated")))
            if reason:
                if reason not in ("request_cooldown","awaiting_ack","pulse_already_requested"):
                    self._record("brake_rejected",e,now,reason=reason,request_id=None)
                    self.last_result=dict(stage="brake_rejected",reason=reason,event_id=e["event_id"],control_mode=self.mode,executed=False)
                continue
            candidates.append(e)
        if not candidates:return None
        request_id=uuid.uuid4().hex
        request=dict(type="action",action="brake",source="hazard",request_id=request_id,
            timestamp_ns=now,events=[dict(event_id=e["event_id"],kind=e["kind"],severity=e["severity"],track_id=e["track_id"]) for e in candidates])
        if any(e["kind"]=="CROSSWALK_BRAKE_PULSE" for e in candidates):request["duration_ms"]=300
        try:
            self.sock.sendto(json.dumps(request,separators=(",",":")).encode(),self.address)
            self.pending=dict(request_id=request_id,sent_ns=now,events=candidates)
            for e in candidates:
                self.last_send[e["event_id"]]=now
                if e["kind"]=="CROSSWALK_BRAKE_PULSE":self.pulse_sent.add(e["event_id"])
                self._record("brake_requested",e,now,request_id=request_id,reason="eligible",executed=False,simulated=self.mode=="DRY_RUN")
            self._poll(now)
            self.last_send={k:v for k,v in self.last_send.items() if now-v<2000000000}
            return self.last_result
        except OSError as exc:
            for e in candidates:self._record("brake_rejected",e,now,request_id=request_id,reason=type(exc).__name__)
            return None

    def close(self):self.sock.close()
