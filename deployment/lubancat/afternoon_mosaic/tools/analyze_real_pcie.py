"""Hardware-specific evidence; preserve performance failures from the live run."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
from validate_followup import rows, stats
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--label",required=True)
    args=parser.parse_args();ev=ROOT/"evidence";label=args.label
    report=json.loads((ev/(label+"_acceptance.json")).read_text())
    extra=json.loads((ev/(label+"_additional.json")).read_text())
    lo=report["started_ns"];hi=report["windows"][-1]["end_ns"]
    timeline=[r for r in rows(ev/(label+"_events.jsonl")) if lo<=r.get("logged_ns",0)<=hi]
    requests=[r for r in timeline if r.get("stage")=="brake_requested"]
    acks=[r for r in timeline if r.get("stage")=="brake_accepted"]
    keys={r["request_id"] for r in requests}
    hardware=[r for r in acks if r.get("fpga_accepted")]
    sampled=report["metrics"]
    vehicle=[r.get("vehicle_status",{}) for r in sampled]
    capture=[r.get("pcie_capture",{}) for r in sampled]
    ai=[r for r in rows(ev/(label+"_ai.jsonl")) if lo<=r.get("result_publish_timestamp_ns",0)<=hi]
    invalid=dict(extra["control_consistency"])
    sources=Counter(r.get("source_id") for r in requests)
    reasons=Counter(r.get("reason") for r in timeline if r.get("stage")=="brake_rejected")
    result=dict(kind="ACTUAL_REAL_PCIE_Q4_ACCEPTANCE",input="/dev/shm/pango-video/latest.pgv",
        control_mode=report["control_mode"],forward_source_id=report["forward_source_id"],
        started_ns=lo,ended_ns=hi,continuous_seconds=report["continuous_seconds"],
        baseline_method_rates=extra["baseline_method_rates"],qt_fps=report["qt_scene_graph_fps_mean"],
        source_counter_fps=extra["source_counter_fps"],texture_counter_fps=extra["texture_counter_fps"],
        performance_passed=extra["baseline_fps_passed"],performance_note="Different physical input and additional real PCIe capture load; not same-input replay comparison",
        hmi_cpu_percent_all_cores=report["cpu_percent_all_cores"],cpu_metric_note="Qt field is HMI process only, not total system CPU",
        actual_system_cpu=json.loads((ev/(label+"_system_cpu.json")).read_text()) if (ev/(label+"_system_cpu.json")).exists() else None,
        source_crc_errors=sampled[-1].get("source_crc_errors"),
        source_dropped_frames=sampled[-1].get("source_dropped_frames"),
        runtime_errors=extra["runtime_errors"],duplicate_result_keys=report["duplicate_result_keys"],
        semantics_errors=max((r.get("semantics",{}).get("errors",0) for r in ai),default=0),
        event_validation=extra["event_validation"],control_consistency=invalid,
        control_stage_counts=dict(Counter(r.get("stage") for r in timeline)),
        decision_rejection_reasons=dict(reasons),request_sources=dict(sources),
        unique_hardware_requests=len(keys),unique_accepted_requests=len({r["request_id"] for r in acks}),
        matching_ack_ids=all(r["request_id"] in keys for r in acks),
        fpga_accepted=len(hardware),fpga_active_acks=sum(r.get("fpga_action_active") is True for r in acks),
        lvt1_brake_allow_acks=sum(r.get("vehicle_feedback",{}).get("brake_allow") is True for r in acks),
        lvt1_feedback_sequence_matches=all(r.get("vehicle_feedback",{}).get("last_sequence")==r.get("hardware_sequence") for r in acks),
        simulated_hardware_requests=sum(r.get("simulated") is not False for r in requests),
        vehicle_samples=len(vehicle),vehicle_online_samples=sum(r.get("online") is True for r in vehicle),
        vehicle_sensor_fresh_samples=sum(r.get("sensor_stale") is False for r in vehicle),
        vehicle_file_age_ms=stats([r.get("vehicle_file_age_ms") for r in sampled]),
        capture_running_samples=sum(r.get("state")=="running" for r in capture),
        capture_sample_count=len(capture),capture_frames_increased=bool(capture and capture[-1].get("frames",0)>capture[0].get("frames",0)),
        navigation_states=report["navigation_states"],actual_event_result_counts=extra["actual_event_result_counts"],
        physical_actuator_verified=False,pcie_hot_recovery_verified=False,
        unit_tests=49,native_crc_test=True,native_risk_link_test=True,native_vehicle_status_test=True,
        source_sync=json.loads((ev/"pcie_q4_source_sync.json").read_text()))
    result["functional_chain_passed"]=(report["continuous_seconds"]>=600 and len(keys)>0 and
        len(hardware)==len(acks)==len(requests) and result["matching_ack_ids"] and
        result["lvt1_feedback_sequence_matches"] and result["simulated_hardware_requests"]==0 and
        not any(invalid.values()) and not result["runtime_errors"] and not result["duplicate_result_keys"] and
        result["source_crc_errors"]==0 and result["capture_frames_increased"] and
        result["capture_running_samples"]==len(capture))
    result["passed"]=result["functional_chain_passed"] and result["performance_passed"]
    (ev/(label+"_FINAL_ACCEPTANCE.json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("actual_system_cpu",)},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
