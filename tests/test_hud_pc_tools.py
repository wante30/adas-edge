import json

from tools.export_adas_status import export_log
from tools.hud_control_demo import run_demo
from competition_io.adas_status_v1 import Action, AdasStatus, Light, SIZE, Valid


def test_export_uses_conservative_source_time_and_never_promotes_unverified_green(tmp_path):
    source = tmp_path / "predictions.jsonl"
    rows = []
    for frame, timestamp, fresh in [(0, 0., True), (1, .033, False), (3, .1, True)]:
        rows.append({"frame": frame, "source_s": timestamp, "main_fresh": fresh,
                     "main_source_frame": frame, "main_age_ms": 0.,
                     "specialist_source_frame": 0, "specialist_age_ms": timestamp*1000,
                     "signal": {"state": "GREEN", "raw_state": "GREEN", "confidence": .8}})
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = export_log(source, tmp_path / "export")
    binary = (tmp_path / "export/official_video_adst.bin").read_bytes()
    assert report["packet_count"] == 2 and len(binary) == 2*SIZE
    last = AdasStatus.deserialize(binary[SIZE:])
    assert last.source_timestamp_ns == 0 and last.source_frame_id == 0
    assert last.light == Light.GREEN and last.action == Action.UNKNOWN
    assert not last.valid & (Valid.DECISION | Valid.SIGNAL_ASSOCIATED)
    assert report["expiry_verified"]


def test_temporal_filter_hold_does_not_refresh_old_observation(tmp_path):
    source = tmp_path / "held.jsonl"
    rows = []
    for frame, timestamp, raw in [(0, 0., 'GREEN'), (3, .1, 'UNKNOWN'), (6, .2, 'RED')]:
        rows.append({"frame": frame, "source_s": timestamp, "main_fresh": True,
                     "main_source_frame": frame, "main_age_ms": 0.,
                     "specialist_source_frame": frame, "specialist_age_ms": 0.,
                     "signal": {"state": "GREEN", "raw_state": raw, "confidence": .8}})
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = export_log(source, tmp_path / "held_export", ttl_ms=150)
    records = [json.loads(line) for line in (tmp_path / "held_export/official_video_adst.jsonl").read_text().splitlines()]
    assert report["temporal_hold_packets_preserving_original_time"] == 2
    assert records[-1]["status"]["source_timestamp_ns"] == 0
    assert records[-1]["receiver"]["reason"] == 'expired_or_clock_rollback'


def test_reference_demo_runs_all_vectors_and_timeout_recovery(tmp_path):
    report = run_demo(tmp_path)
    assert report["passed"] and len(report["vectors"]) == 9
    assert report["timeout"]["phase"] == "UNKNOWN" and report["resume"]["applied"]
    assert report["covered"]["covered"] and report["release"]["applied"]
    assert (tmp_path / "hud_frame_trace.csv").is_file()
