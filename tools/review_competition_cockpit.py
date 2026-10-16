"""Re-render recorded video observations and explicit software rule fixtures.

No models run here. Real observations are read from a previously saved PC trace;
rule fixtures are displayed on a plain background, never on a real road frame.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from competition_ai.bound_signal import SignalObservation, classify_horizontal_head
from competition_ai.hazard_events import HazardObservation
from competition_ai.image_motion import ImageMotion
from competition_ai.scenario import Boundary, Controls, Direction, Lane, Light, Maneuver, Mode, Motion, Perception
from competition_ai.scenario_policy import decide
from tools.competition_console.pipeline import FrontSnapshot
from tools.competition_console.view import NativeView


DEFAULT_TRACE = ROOT / "outputs/hazard_events_20261014/native_final/trace.jsonl"
OFFICIAL_VIDEO_SHA256 = "b91c2bc464189e125e69ace24024d19a3870a5c7bf97d750bbc7fcc7ca2b4ea7"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size):
    paths = (Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    candidate = next((p for p in paths if p.exists()), None)
    return ImageFont.truetype(str(candidate), size) if candidate else ImageFont.load_default(size=size)


def observation(row):
    data = dict(row["signal"])
    data["light"] = Light(data["light"])
    data["raw_light"] = Light(data["raw_light"])
    if data["bbox"] is not None:
        data["bbox"] = tuple(data["bbox"])
    hazards = dict(row.get("hazards", {}))
    if hazards:
        hazards["events"] = tuple(hazards.get("events", ()))
        hazards["tracks"] = tuple(hazards.get("tracks", ()))
        hazards["motion"] = ImageMotion(**hazards.get("motion", {}))
    return FrontSnapshot(row["main"], row["specialist"], SignalObservation(**data),
                         row.get("main_age_ms"), row.get("specialist_age_ms"),
                         row["binding_source"], HazardObservation(**hazards))


def fixture_frame(title, lines):
    image = Image.new("RGB", (1920, 1080), "#18232f")
    draw = ImageDraw.Draw(image)
    draw.text((960, 300), "软件规则输入，非视频识别", font=font(50), fill="#ffc577", anchor="mm")
    draw.text((960, 410), title, font=font(62), fill="#f1f5f7", anchor="mm")
    for index, line in enumerate(lines):
        draw.text((960, 550 + index * 70), line, font=font(36), fill="#91a2b3", anchor="mm")
    draw.text((960, 840), "仅验证状态机、界面和报警；不证明模型已经识别该事件。",
              font=font(29), fill="#91a2b3", anchor="mm")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def expected(maneuver, direction, allowed, alarms, motion, hint=Direction.NONE):
    return {"maneuver": maneuver.value, "direction": direction.value, "request_allowed": allowed,
            "alarms": list(alarms), "simulated_motion": motion.value, "navigation_hint": hint.value,
            "led_mask": {Direction.NONE: 0, Direction.LEFT: 1, Direction.FORWARD: 2, Direction.RIGHT: 4}[direction]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; existing paths are rejected")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--video", type=Path, help="Optional relocated copy of the exact official video")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new directory: {output}")
    trace_path = args.trace.resolve()
    rows = {int(row["frame"]): row for row in
            (json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    red, green = rows[363], rows[1893]
    video = args.video.resolve() if args.video else Path(red["video"])
    video_hash = sha256(video)
    if video_hash != OFFICIAL_VIDEO_SHA256:
        parser.error("Video does not match the verified official source SHA256; cached detections cannot be reused")
    if red["video"] != green["video"]:
        parser.error("Selected cached observations do not share a source video")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        parser.error(f"Could not open official video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    real_inputs = {}
    for row, color in ((red, Light.RED), (green, Light.GREEN)):
        fid = row["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (fid + 1)) > .1:
            parser.error(f"Cannot decode the exact official frame {fid}")
        snap = observation(row)
        pixel_color = classify_horizontal_head(frame, snap.signal.bbox)
        if snap.signal.light != color or pixel_color != color or not snap.signal.associated:
            parser.error(f"Cached/actual lamp color mismatch at frame {fid}")
        if abs(row["source_s"] - fid / fps) > 1e-6:
            parser.error(f"Cached source timestamp does not match video frame {fid}")
        real_inputs[color] = (frame, snap, row)
    cap.release()
    output.mkdir(parents=True, exist_ok=False)
    code_names = ("tools/competition_console/view.py", "tools/competition_console/ego_asset.py",
                  "competition_ai/scenario.py", "competition_ai/scenario_policy.py", "tools/review_competition_cockpit.py")
    code_hashes = {name: sha256(ROOT / name) for name in code_names}
    view = NativeView()
    cases = []

    def run_case(case_id, title, controls, perception, target, *, real=None, lines=()):
        if real is not None:
            frame, snap, row = real_inputs[real]
            now_ns = round(row["source_s"] * 1e9)
            perception = snap.as_perception()
            fid = row["frame"]
            evidence = "VIDEO"
            source = {"kind": "RECORDED_PC_VIDEO_OBSERVATIONS", "video": str(video), "trace": str(trace_path),
                      "frame_id": fid, "source_s": row["source_s"], "signal_observation": asdict(snap.signal),
                      "binding_source": snap.binding_source, "main_count": len(snap.main),
                      "specialist_count": len(snap.specialist), "inference_rerun": False,
                      "controls_origin": "SOFTWARE_SIMULATION", "lamp_binding_origin": "EXPLICIT_SCENE_PROFILE",
                      "pixel_color_recheck": real.value}
            frame_count = total_frames
        else:
            now_ns = 10_000_000_000
            fid = 300
            evidence = "SOFTWARE_RULE_FIXTURE"
            frame = fixture_frame(title, lines)
            snap = FrontSnapshot([], [], SignalObservation(light=perception.light, raw_light=perception.light,
                                  timestamp_ns=perception.signal_timestamp_ns, associated=perception.signal_associated,
                                  reason="software_rule_fixture", source="SOFTWARE_RULE_FIXTURE"),
                                  None, None, "SOFTWARE_RULE_FIXTURE")
            source = {"kind": "SOFTWARE_RULE_FIXTURE", "video": None, "visual_recognition": False,
                      "background": "PLAIN_COLOR_WITH_EXPLICIT_NON_VIDEO_LABEL", "controls_origin": "SOFTWARE_SIMULATION"}
            frame_count = 300
        decision = decide(controls, perception, now_ns)
        actual = {**asdict(decision), "led_mask": decision.led_mask}
        actual = json.loads(json.dumps(actual, ensure_ascii=False))
        checks = {name: actual[name] == value for name, value in target.items()}
        pixels = view.render(frame, snap, controls, decision, frame_id=fid, fps=fps, paused=True,
                             total_frames=frame_count, source_name=str(video) if real else "软件规则输入，非视频识别",
                             evidence=evidence, perception=perception, transport_status="PCIe 未连接")
        image_path = output / f"{case_id}.png"
        if not cv2.imwrite(str(image_path), pixels):
            raise OSError(f"Failed to write {image_path}")
        checks["canvas_1600x900"] = tuple(pixels.shape[:2]) == (900, 1600)
        checks["nonblank_render"] = float(pixels.std()) > 10
        cases.append({"id": case_id, "title": title, "source": source, "controls": asdict(controls),
                      "perception": asdict(perception), "now_ns": now_ns, "expected": target,
                      "actual": actual, "checks": checks, "passed": all(checks.values()), "image": str(image_path)})

    run_case("01_video_red_wait", "真实视频：红灯待行", Controls(), None,
             expected(Maneuver.STOP, Direction.NONE, False, (), Motion.STOP), real=Light.RED)
    run_case("02_video_red_forced_go", "真实视频：红灯强制行驶报警", Controls(motion=Motion.GO), None,
             expected(Maneuver.STOP, Direction.NONE, False, ("RED_LIGHT",), Motion.GO), real=Light.RED)
    run_case("03_video_green_straight", "真实视频：绿灯直行", Controls(), None,
             expected(Maneuver.FORWARD, Direction.FORWARD, True, (), Motion.GO), real=Light.GREEN)
    run_case("04_video_green_left", "真实视频：绿灯左转车道", Controls(lane=Lane.LEFT), None,
             expected(Maneuver.TURN_LEFT, Direction.LEFT, True, (), Motion.GO), real=Light.GREEN)
    now = 10_000_000_000
    run_case("05_rule_solid_forbidden", "实线强制变道报警",
             Controls(mode=Mode.ROAD, request=Maneuver.CHANGE_LEFT),
             Perception(left_boundary=Boundary.SOLID, lane_timestamp_ns=now),
             expected(Maneuver.FORWARD, Direction.FORWARD, False, ("SOLID_LANE",), Motion.GO),
             lines=("输入：左边界 = 实线；请求 = 向左变道", "预期：保持直行，拒绝变道并报警"))
    run_case("06_rule_dashed_allowed", "虚线允许变道",
             Controls(mode=Mode.ROAD, request=Maneuver.CHANGE_RIGHT),
             Perception(right_boundary=Boundary.DASHED, lane_timestamp_ns=now),
             expected(Maneuver.CHANGE_RIGHT, Direction.RIGHT, True, (), Motion.GO),
             lines=("输入：右边界 = 虚线；请求 = 向右变道", "预期：允许向右变道，点亮右向指示"))
    run_case("07_rule_unknown_boundary", "未知边界拒绝变道",
             Controls(mode=Mode.ROAD, request=Maneuver.CHANGE_LEFT), Perception(),
             expected(Maneuver.FORWARD, Direction.FORWARD, False, ("UNKNOWN_LANE",), Motion.GO),
             lines=("输入：车道线未确认；请求 = 向左变道", "预期：保持直行，拒绝未知边界变道"))
    run_case("08_rule_navigation_missing", "导航缺失等待",
             Controls(mode=Mode.NAVIGATION), Perception(),
             expected(Maneuver.STOP, Direction.NONE, False, ("NO_NAVIGATION",), Motion.STOP),
             lines=("输入：导航指令无效", "预期：等待有效导航，禁止默认为成功导航"))
    run_case("09_rule_navigation_hint", "有效导航提示不提前转",
             Controls(mode=Mode.NAVIGATION, navigation=Direction.LEFT, navigation_valid=True), Perception(),
             expected(Maneuver.FORWARD, Direction.FORWARD, True, (), Motion.GO, Direction.LEFT),
             lines=("输入：有效路线提示 = 左转；当前无即时转向请求", "预期：显示左转提示，当前保持直行"))
    run_case("10_rule_pedestrian_priority", "行人危险优先制动",
             Controls(mode=Mode.HAZARD, signalized=True),
             Perception(light=Light.GREEN, signal_associated=True, signal_timestamp_ns=now,
                        pedestrian_hazard=True, hazard_timestamp_ns=now),
             expected(Maneuver.BRAKE, Direction.NONE, False, ("PEDESTRIAN_DANGER",), Motion.STOP),
             lines=("输入：新鲜行人危险事件；控制灯 = 绿色", "预期：危险优先，立即制动并报警", "不代表已经识别行人闯红灯"))
    run_case("11_rule_unknown_signal", "未知信号禁止通行", Controls(),
             Perception(signal_associated=True, signal_timestamp_ns=now),
             expected(Maneuver.STOP, Direction.NONE, False, ("UNKNOWN_SIGNAL",), Motion.STOP),
             lines=("输入：控制灯已经关联，但灯色未知", "预期：等待灯色，禁止通行"))
    run_case("12_rule_expired_signal", "过期信号禁止通行", Controls(),
             Perception(light=Light.GREEN, signal_associated=True, signal_timestamp_ns=now - 1_000_000_000),
             expected(Maneuver.STOP, Direction.NONE, False, ("STALE_SIGNAL",), Motion.STOP),
             lines=("输入：最后一次绿灯观测已超过有效期", "预期：绿灯旧结果不可继续放行"))

    sheet = Image.new("RGB", (1600, 4 * 348 + 96), "#0b1118")
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 18), "车机状态验收：4 个视频回放状态 + 8 个独立软件规则用例", font=font(27), fill="#f1f5f7")
    draw.text((24, 58), "PC 缓存观测回放与规则检查；未执行板端、PCIe 或 FPGA 验证。", font=font(20), fill="#91a2b3")
    for index, case in enumerate(cases):
        x, y = (index % 3) * 532 + 8, (index // 3) * 348 + 96
        with Image.open(case["image"]) as image:
            sheet.paste(image.resize((520, 293), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 4, y + 300), ("PASS  " if case["passed"] else "FAIL  ") + case["title"],
                  font=font(18), fill="#80d5cf" if case["passed"] else "#ff655e")
    sheet.save(output / "contact_sheet.png")
    changed_code = [name for name, digest in code_hashes.items() if sha256(ROOT / name) != digest]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "PC_CACHED_OBSERVATION_AND_RULE_UI_REVIEW",
              "no_inference_rerun": True, "no_board_or_fpga_verification": True,
              "caveats": ["Video detections and lamp observations come from the existing PC trace, not a new inference run.",
                          "Simulated lane selection and motion buttons do not change the vehicle in the original video.",
                          "Lane-to-lamp association is provided by an explicit scene profile, not inferred automatically.",
                          "Eight rule fixtures do not demonstrate lane, navigation, or hazard recognition accuracy."],
              "source_video": str(video), "source_video_sha256": video_hash, "source_trace": str(trace_path),
              "source_trace_sha256": sha256(trace_path), "fps": fps, "source_total_frames": total_frames,
              "code_sha256": code_hashes, "code_changed_during_review": changed_code,
              "real_video_cases": 4, "software_rule_cases": 8, "passed": not changed_code and all(case["passed"] for case in cases),
              "cases": cases, "contact_sheet": str(output / "contact_sheet.png")}
    (output / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "cases": len(cases), "output": str(output)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
