#!/usr/bin/env python3
"""Validate accounting and hardware evidence in a Qt GPU probe report."""

import json
import math
import sys


def main(path, sidecar_path=None):
    with open(path, "r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    produced = report["produced_frames"]
    consumed = report["successfully_uploaded_frames"]
    not_uploaded = report["source_frames_not_uploaded"]
    duration = report["duration_s"]
    assert produced == consumed + not_uploaded, (
        f"frame accounting mismatch: produced={produced}, uploaded={consumed}, "
        f"not_uploaded={not_uploaded}"
    )
    assert consumed == report["consumed_frames"]
    assert consumed > 0, "no RGB565 texture upload completed"
    expected_fps = consumed / duration
    actual_fps = report["unique_source_fps_average"]
    assert math.isclose(actual_fps, expected_fps, rel_tol=0.01), (
        f"average unique source FPS mismatch: reported={actual_fps}, "
        f"expected={expected_fps}"
    )
    assert report["hardware_accelerated"] is True
    assert "Mali-G52" in report["gl_renderer"]
    assert report["process_rss_mb_last"] > 0
    assert any("libmali.so" in path for path in report["loaded_graphics_libraries"])
    assert report["physical_scanout_fps"] is None
    unique_replay_frames = report["unique_replay_content_frames_uploaded"]
    source = report["frame_source"]
    if source in ("generated_rgb565_pattern", "pgv-live", "pgv-fixture"):
        # Generated frames and live PGVF snapshots use the producer frame ID
        # as their content index; there is no replay manifest to bound this.
        assert unique_replay_frames == consumed
    else:
        assert 0 <= unique_replay_frames <= report["replay_manifest_frame_count"]
    assert report["unique_replay_content_fps_average"] <= actual_fps + 0.01
    if source == "board_rgb565_replay":
        expected_wrap = produced > report["replay_manifest_frame_count"]
        assert report["replay_content_wrapped"] is expected_wrap

    if sidecar_path:
        with open(sidecar_path, "r", encoding="utf-8") as sidecar_file:
            sidecar = json.load(sidecar_file)
        assert sidecar["input_requests_received"] == report["ai_requests_sent"]
        assert sidecar["shared_frame_identity_mismatches"] == 0
        assert sidecar["inference_errors"] == 0
        assert sidecar["result_packets_sent"] > 0
        assert report["ai_results_received"] > 0
        updates = sidecar["updates_per_second"]
        assert updates["yolo11n_rknn"] > 0
        assert updates["traffic_light_specialist_rknn"] > 0
        assert updates["navigation_ocr"] > 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            f"usage: {sys.argv[0]} REPORT.json [AI_SIDECAR_SUMMARY.json]"
        )
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
