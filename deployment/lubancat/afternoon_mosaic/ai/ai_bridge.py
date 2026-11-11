#!/usr/bin/env python3
"""Latest-only RKNN/OCR sidecar for the Mali RGB565 renderer."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import json
import math
import multiprocessing
import os
import queue
import signal
import traceback
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time

try:
    import resource
except ImportError:
    resource = None

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_configured_root = os.environ.get("ADAS_EDGE_ROOT")
ROOT = Path(_configured_root).resolve() if _configured_root else Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_ai.scenario import Lane
from tools.competition_console.fpga_perception import TrafficCode, build_hud_perception
from auto_brake import AutoBrakeController
from light_semantics import SemanticWorker
from light_risk import LightRiskEngine
from primary_tracks import PrimaryTracks
from risk_events import EventStore, RiskObservation
from q2_navigation_reader import CARD_ROI, Q2NavigationReader
from navigation_gate import NavigationGate
from tools.competition_console.navigation_reader import NavigationObservation
from tools.competition_console.navigation_reader import NavigationObservation
from tools.competition_console.pipeline import FrontPipeline
from tools.competition_console.rknn_detector import RknnConsoleDetector
from tools.competition_console.latest_frame_slot import LatestFrameSlot
from model_profiles import (
    DEFAULT_MODEL_ID,
    build_model_profiles,
    default_main_model_path,
    decode_model_switch_request,
    encode_model_switch_reply,
)


WIDTH = 1280
HEIGHT = 720
FRAME_BYTES = WIDTH * HEIGHT * 2
REQUEST_HEADER = struct.Struct("<4sQQ")
REQUEST_HEADER_V2 = struct.Struct("<4sQQff")
RESULT_HEADER = struct.Struct("<4sQQIIB3xf")
BOX = struct.Struct("<7f")
RESULT_HEADER_V2 = struct.Struct("<4sQQIIB3xfI")
BOX_V2 = struct.Struct("<7fIf")
MAX_BOXES = 32
DYNAMIC_HMI_WIDTH = 238
DYNAMIC_HMI_HEIGHT = 452
DYNAMIC_HMI_HEARTBEAT_NS = 500_000_000
DYNAMIC_RESULT_TTL_MS = 250.0
DYNAMIC_HMI_HEADER = struct.Struct("<4sHHQQ")
FRAME_SNAPSHOT_HEADER = struct.Struct("<4sQQII")
FLAG_RED = 1 << 0
FLAG_GREEN = 1 << 1
FLAG_PEDESTRIAN_HAZARD = 1 << 2
FLAG_OBSTACLE_HAZARD = 1 << 3
FLAG_NAVIGATION_VALID = 1 << 4
NAV_DIRECTIONS = {"none": 0, "left": 1, "right": 2, "forward": 3, "uturn": 4}
CLASS_IDS = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "traffic light": 9,
}


def map_detection_box(bbox, *, input_layout, display_view, road_quadrant):
    if input_layout not in ("quad", "front", "full") or display_view not in ("quad", "front"):
        raise ValueError("input_layout must be quad, front, or full")
    if road_quadrant not in (0, 1, 2, 3):
        raise ValueError("road_quadrant must be 0, 1, 2, or 3")
    values = tuple(float(value) for value in bbox)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("bbox must contain four finite values")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")
    if input_layout == "full":
        return values
    if input_layout == "front":
        if display_view == "quad":
            raise ValueError("A single-front source cannot display four camera views")
        return values
    offset_x, offset_y = _quadrant_offset(road_quadrant)
    if display_view == "quad":
        return x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
    return x1 * 2.0, y1 * 2.0, x2 * 2.0, y2 * 2.0


def build_dynamic_status(*, source, frame_id, timestamp_ns, now_ns, hud, navigation,
                         hazard_timestamp_ns, hazards, detections, signal_age_ms,
                         main_age_ms, specialist_age_ms, navigation_ttl_ms=3000,
                         source_stopped=False, source_available=True, canonical_risks=False,
                         input_fps=None, submitted_fps=None, ai_fps=None):
    traffic_reason = str(hud.get("traffic_evidence") or "UNKNOWN")
    traffic = str(hud.get("traffic") or "OFF").upper()
    if source_stopped or not source_available:
        traffic = "UNKNOWN"
        signal_age_ms = None
    elif "YELLOW" in traffic_reason:
        traffic = "YELLOW"
    elif traffic not in ("RED", "GREEN"):
        traffic = "UNKNOWN"

    nav_source = navigation.get("source_frame_id")
    nav_timestamp = navigation.get("timestamp_ns")
    age_ms = ((now_ns - nav_timestamp) / 1_000_000
              if isinstance(nav_timestamp, int) and 0 <= nav_timestamp <= now_ns else None)
    nav_identity_valid = (
        type(nav_source) is int and 0 <= nav_source <= frame_id
        and type(nav_timestamp) is int and 0 <= nav_timestamp <= timestamp_ns <= now_ns
    )
    nav_fresh = age_ms is not None and age_ms < navigation_ttl_ms
    ocr_timestamp = navigation.get("ocr_timestamp_ns")
    ocr_age_ms = ((now_ns - ocr_timestamp) / 1_000_000
                  if isinstance(ocr_timestamp, int) and 0 <= ocr_timestamp <= now_ns else None)
    ocr_fresh = ocr_age_ms is not None and ocr_age_ms < navigation_ttl_ms
    nav_valid = (bool(navigation.get("valid")) and nav_identity_valid
                 and nav_fresh and ocr_fresh and not source_stopped)
    if source_stopped:
        nav_state = "STOPPED"
    elif navigation.get("valid") and nav_identity_valid and (not nav_fresh or not ocr_fresh):
        nav_state = "STALE"
    elif nav_valid and navigation.get("direction") in NAV_DIRECTIONS:
        nav_state = "VALID"
    else:
        nav_state = "UNREADABLE"
    nav = {
        "state": nav_state,
        "direction": navigation.get("direction", "none") if nav_state == "VALID" else "none",
        "distance_m": navigation.get("distance_m") if nav_state == "VALID" else None,
        "road_name": str(navigation.get("road_name") or "") if nav_state == "VALID" else "",
        "source_frame_id": nav_source,
        "age_ms": age_ms,
        "ocr_timestamp_ns": ocr_timestamp,
        "ocr_age_ms": ocr_age_ms,
        "reason": navigation.get("reason", ""),
        "valid": nav_state == "VALID",
    }

    hazard_age_ms = ((now_ns - hazard_timestamp_ns) / 1_000_000
                     if isinstance(hazard_timestamp_ns, int)
                     and 0 <= hazard_timestamp_ns <= now_ns else None)
    hazard_kinds = []
    hazard_details = []
    if (not source_stopped and source_available and hazard_age_ms is not None
            and hazard_age_ms < (400.0 if canonical_risks else DYNAMIC_RESULT_TTL_MS)):
        for item in hazards:
            if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
                continue
            if canonical_risks and (not item.get("valid") or item.get("lifecycle_state")!="ACTIVE" or item.get("expires_at_ns",0)<=now_ns):continue
            kind = item["kind"]
            hazard_kinds.append(kind)
            severity = item.get("severity", "warning")
            if severity not in ("info", "warning", "critical"):
                severity = "warning"
            detail = {
                "event_type": kind,
                "source_id": source,
                "source_frame_id": int(frame_id),
                "timestamp": int(hazard_timestamp_ns),
                "severity": severity,
                "valid": True,
                "ttl_ms": DYNAMIC_RESULT_TTL_MS,
                "evidence": item.get("source") or "IMAGE_PERCEPTION",
            }
            track_id = item.get("track_id")
            if isinstance(track_id, int) and not isinstance(track_id, bool) and track_id > 0:
                detail["track_id"] = track_id
            bbox = item.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    values = [float(value) for value in bbox]
                except (TypeError, ValueError):
                    values = None
                if values is not None and all(math.isfinite(value) for value in values):
                    detail["bbox"] = values
            target_class = item.get("target_class") or item.get("class_name")
            if isinstance(target_class, str) and target_class:
                detail["target_class"] = target_class
            if item.get("event_id"):
                detail = dict(item)  # Preserve the canonical identity/lifecycle for every consumer.
            hazard_details.append(detail)
    if (not canonical_risks and not source_stopped and source_available and traffic == "RED"
            and signal_age_ms is not None and signal_age_ms < 2500):
        hazard_kinds.append("RED_SIGNAL_CONFIRMED")
        hazard_details.append({
            "event_type": "RED_SIGNAL_CONFIRMED",
            "source_id": source,
            "source_frame_id": int(frame_id),
            "timestamp": int(timestamp_ns),
            "severity": "info",
            "valid": True,
            "ttl_ms": 2500,
            "evidence": traffic_reason,
        })
    hazard_kinds = list(dict.fromkeys(hazard_kinds))

    result_age_ms = ((now_ns - timestamp_ns) / 1_000_000
                     if source_available and not source_stopped
                     and 0 <= timestamp_ns <= now_ns else None)
    counts = {}
    if (not source_stopped and source_available and result_age_ms is not None
            and result_age_ms < DYNAMIC_RESULT_TTL_MS):
        for item in detections:
            if not isinstance(item, dict):
                continue
            name = str(item.get("class_name") or "unknown")
            counts[name] = counts.get(name, 0) + 1
    return {
        "source": str(source),
        "frame_id": int(frame_id),
        "timestamp_ns": int(timestamp_ns),
        "traffic": traffic,
        "traffic_reason": traffic_reason,
        "signal_age_ms": signal_age_ms,
        "result_age_ms": result_age_ms,
        "navigation": nav,
        "hazards": hazard_kinds,
        "hazard_details": hazard_details,
        "hazard_age_ms": hazard_age_ms,
        "detections": counts,
        "main_age_ms": None if source_stopped or not source_available else main_age_ms,
        "specialist_age_ms": None if source_stopped or not source_available else specialist_age_ms,
        "input_fps": None if source_stopped or not source_available else input_fps,
        "submitted_fps": None if source_stopped or not source_available else submitted_fps,
        "ai_fps": None if source_stopped or not source_available else ai_fps,
        "runtime_status": "SOURCE_STOPPED" if source_stopped else "RUNNING",
        "pcie_state": ("CONNECTED" if source_available and not source_stopped else "DISCONNECTED")
                      if source == "PGV_LIVE" else "NOT_LIVE",
        "fpga_perception": ("NOT_EXPOSED_BY_FRAME_SOURCE" if source_available and not source_stopped
                            else "UNKNOWN") if source == "PGV_LIVE" else "NOT_CONNECTED",
    }


def initial_dynamic_status(source, now_ns):
    status = build_dynamic_status(
        source=source,
        frame_id=0,
        timestamp_ns=now_ns,
        now_ns=now_ns,
        hud={"traffic": "OFF", "traffic_evidence": "NO_DATA"},
        navigation={},
        hazard_timestamp_ns=None,
        hazards=(),
        detections=(),
        signal_age_ms=None,
        main_age_ms=None,
        specialist_age_ms=None,
        source_available=False,
    )
    status["runtime_status"] = "WAITING_FOR_SOURCE"
    return status


def timing_detection_counts(status):
    counts = status.get("detections") if isinstance(status, dict) else None
    if not isinstance(counts, dict):
        return {}
    return {
        str(name): int(count)
        for name, count in counts.items()
        if isinstance(name, str) and name and isinstance(count, int)
        and not isinstance(count, bool) and count >= 0
    }


def should_refresh_stopped_status(now_ns, last_activity_ns, last_refresh_ns,
                                  heartbeat_ns=DYNAMIC_HMI_HEARTBEAT_NS):
    if now_ns < last_activity_ns or now_ns - last_activity_ns < heartbeat_ns:
        return False
    return last_refresh_ns is None or now_ns - last_refresh_ns >= heartbeat_ns


def _font(size, font_path=None):
    candidates = [font_path] if font_path else []
    candidates.extend((
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ))
    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def hmi_text_width(draw, text, font):
    if hasattr(draw, "textbbox"):
        bounds = draw.textbbox((0, 0), text, font=font)
        return bounds[2] - bounds[0]
    return draw.textsize(text, font=font)[0]


def fit_hmi_text(draw, text, font, max_width):
    text = str(text)
    if hmi_text_width(draw, text, font) <= max_width:
        return text
    suffix = "..."
    while text and hmi_text_width(draw, text + suffix, font) > max_width:
        text = text[:-1]
    return text.rstrip() + suffix


def render_dynamic_hmi_rgba(status, font_path=None):
    image = Image.new("RGBA", (DYNAMIC_HMI_WIDTH, DYNAMIC_HMI_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (0, 0, DYNAMIC_HMI_WIDTH - 1, DYNAMIC_HMI_HEIGHT - 1),
        fill=(9, 18, 25, 255),
        outline=(67, 186, 179, 255),
        width=1,
    )
    font_title, font_value, font_small = _font(16, font_path), _font(14, font_path), _font(11, font_path)
    text_width = DYNAMIC_HMI_WIDTH - 36
    ink, muted, teal, red, amber, green = "#f1f5f7", "#91a2b3", "#80d5cf", "#ff655e", "#ffc577", "#8ed47e"
    traffic = {"RED": ("红灯", red), "GREEN": ("绿灯", green),
               "YELLOW": ("黄灯", amber), "UNKNOWN": ("未确认", muted)}
    draw.text((18, 8), fit_hmi_text(draw, "动态感知", font_title, text_width),
              font=font_title, fill=ink)
    source = str(status.get("source", "UNKNOWN"))
    frame_id = status.get("frame_id")
    runtime_names = {
        "RUNNING": "运行中", "WAITING_FOR_SOURCE": "等待视频",
        "SOURCE_STOPPED": "视频已停止",
    }
    source_line = f"{source} / #{frame_id}" if frame_id else f"{source} / 等待首帧"
    draw.text((18, 33), fit_hmi_text(draw, source_line, font_small, text_width),
              font=font_small, fill=amber)
    input_fps, submitted_fps = status.get("input_fps"), status.get("submitted_fps")
    if isinstance(input_fps, (int, float)) and isinstance(submitted_fps, (int, float)):
        video_rate = f"进 {input_fps:.1f} / 提 {submitted_fps:.1f} fps"
    else:
        video_rate = runtime_names.get(status.get("runtime_status"), "视频速率 UNKNOWN")
    draw.text((18, 51), fit_hmi_text(draw, video_rate, font_small, text_width),
              font=font_small, fill=muted)
    pcie_names = {"CONNECTED": "PCIe 已连接", "DISCONNECTED": "PCIe 未连接",
                  "NOT_LIVE": "PCIe 非实时"}
    fpga_names = {"NOT_CONNECTED": "FPGA 未接入", "UNKNOWN": "FPGA 未知",
                  "NOT_EXPOSED_BY_FRAME_SOURCE": "FPGA 感知未接"}
    system_line = (f"{pcie_names.get(status.get('pcie_state'), 'PCIe UNKNOWN')} / "
                   f"{fpga_names.get(status.get('fpga_perception'), 'FPGA UNKNOWN')}")
    draw.text((18, 69), fit_hmi_text(draw, system_line, font_small, text_width),
              font=font_small, fill=muted)
    light_text, light_color = traffic.get(str(status.get("traffic", "UNKNOWN")), traffic["UNKNOWN"])
    draw.ellipse((18, 98, 32, 112), fill=light_color)
    draw.text((42, 94), f"信号灯  {light_text}", font=font_value, fill=light_color)
    reason_names = {
        "NO_DATA": "等待源帧数据",
        "SOURCE_STOPPED": "视频源已停止",
        "NO_SIGNAL_ROI": "未检出信号灯候选",
        "NO_CONFIDENT_SIGNAL_ROI": "候选置信度不足",
        "SIGNAL_COLOR_UNCONFIRMED": "颜色证据不足",
        "CONFLICTING_CONFIRMED_SIGNAL_COLORS": "灯色证据冲突",
        "YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL": "黄灯已识别，HUD不编码",
    }
    reason = reason_names.get(status.get("traffic_reason"), "颜色来源已验证")
    draw.text((18, 119), fit_hmi_text(draw, reason, font_small, text_width),
              font=font_small, fill=muted)

    nav = status.get("navigation") or {}
    nav_state = nav.get("state", "UNREADABLE")
    draw.text((18, 151), fit_hmi_text(draw, "导航", font_title, text_width),
              font=font_title, fill=ink)
    if nav_state == "VALID":
        directions = {"left": "左转", "right": "右转", "forward": "直行", "uturn": "掉头"}
        distance = nav.get("distance_m")
        detail = directions.get(nav.get("direction"), "方向未确认")
        if isinstance(distance, (int, float)) and math.isfinite(float(distance)) and distance >= 0:
            detail = f"{distance:g} 米后 {detail}"
        draw.text((18, 178), fit_hmi_text(draw, detail, font_value, text_width),
                  font=font_value, fill=teal)
        road_name = str(nav.get("road_name") or "")
        if road_name:
            road_line, road_color = f"道路  {road_name}", ink
        else:
            road_line, road_color = "道路名称未识别", muted
        draw.text((18, 206), fit_hmi_text(draw, road_line, font_small, text_width),
                  font=font_small, fill=road_color)
    else:
        message = "导航已停止更新" if nav_state == "STOPPED" else "导航暂不可读"
        draw.text((18, 178), fit_hmi_text(draw, message, font_value, text_width),
                  font=font_value, fill=amber if nav_state == "STOPPED" else muted)
    nav_age = nav.get("age_ms")
    nav_age_line = (f"源帧 {nav.get('source_frame_id', 'UNKNOWN')} / 龄期 {nav_age:.0f} ms"
                    if isinstance(nav_age, (int, float))
                    else f"源帧 {nav.get('source_frame_id', 'UNKNOWN')} / 龄期 UNKNOWN")
    draw.text((18, 232), fit_hmi_text(draw, nav_age_line, font_small, text_width),
              font=font_small, fill=muted)

    hazards = status.get("hazards") or []
    hazard_names = {
        "PEDESTRIAN_CROSSING_PATH": "行人进入预测路径",
        "PEDESTRIAN_IN_PATH": "行人进入前方路径",
        "FRONT_COLLISION_RISK": "前方碰撞风险",
        "VEHICLE_CUT_IN_CANDIDATE": "车辆切入候选",
        "LEAD_RAPID_CLOSING": "前车快速接近",
        "LEAD_BRAKE_LAMP_RISE": "前车制动灯变化",
        "SUSPECTED_BRAKING_WITH_CLOSING": "制动灯变化且快速接近",
    }
    draw.text((18, 277), fit_hmi_text(draw, "危险分析", font_title, text_width),
              font=font_title, fill=ink)
    if hazards:
        for index, kind in enumerate(hazards[:3]):
            hazard_line = hazard_names.get(kind, str(kind))
            draw.text((18, 303 + index * 22),
                      fit_hmi_text(draw, hazard_line, font_small, text_width),
                      font=font_small, fill=red)
    else:
        draw.text((18, 303), fit_hmi_text(draw, "未检出有效危险事件", font_small, text_width),
                  font=font_small, fill=muted)

    # Lane metadata is not part of the current MVA packet. Keep this explicit
    # rather than turning an absent FPGA field into a decorative lane state.
    draw.text((18, 374), fit_hmi_text(draw, "车道状态", font_value, text_width),
              font=font_value, fill=ink)
    draw.text((18, 402), fit_hmi_text(draw, "未接入 / FPGA 元数据 UNKNOWN", font_small, text_width),
              font=font_small, fill=muted)
    ai_fps = status.get("ai_fps")
    result_age = status.get("result_age_ms")
    ai_line = (f"AI {ai_fps:.1f} fps" if isinstance(ai_fps, (int, float))
               else "AI UNKNOWN fps")
    ai_line += (f" / 结果 {result_age:.0f} ms" if isinstance(result_age, (int, float))
                else " / 结果 UNKNOWN")
    draw.text((18, 424), fit_hmi_text(draw, ai_line, font_small, text_width),
              font=font_small, fill=muted)
    main_age = status.get("main_age_ms")
    specialist_age = status.get("specialist_age_ms")
    main_age_text = f"{main_age:.0f}" if isinstance(main_age, (int, float)) else "--"
    specialist_age_text = (f"{specialist_age:.0f}"
                           if isinstance(specialist_age, (int, float)) else "--")
    model_line = f"YOLO {main_age_text} / 灯 {specialist_age_text} ms"
    draw.text((18, 436), fit_hmi_text(draw, model_line, font_small, text_width),
              font=font_small, fill=muted)
    return image.tobytes()


def encode_dynamic_hmi_file(status, pixels):
    expected = DYNAMIC_HMI_WIDTH * DYNAMIC_HMI_HEIGHT * 4
    if len(pixels) != expected:
        raise ValueError("Dynamic HMI layer has an invalid RGBA payload length")
    header = DYNAMIC_HMI_HEADER.pack(
        b"HMI1", DYNAMIC_HMI_WIDTH, DYNAMIC_HMI_HEIGHT,
        int(status.get("frame_id", 0)), int(status.get("timestamp_ns", 0)),
    )
    return header + pixels


def encode_frame_snapshot(frame_id, timestamp_ns, pixels):
    if len(pixels) != FRAME_BYTES:
        raise ValueError("AI source snapshot must contain one packed 1280x720 RGB565 frame")
    header = FRAME_SNAPSHOT_HEADER.pack(
        b"MFR1", int(frame_id), int(timestamp_ns), WIDTH * 2, FRAME_BYTES,
    )
    return header + pixels


def decode_frame_snapshot(payload, expected_frame_id, expected_timestamp_ns):
    if len(payload) != FRAME_SNAPSHOT_HEADER.size + FRAME_BYTES:
        return None
    magic, frame_id, timestamp_ns, stride_bytes, payload_bytes = FRAME_SNAPSHOT_HEADER.unpack_from(payload)
    if (magic, frame_id, timestamp_ns, stride_bytes, payload_bytes) != (
        b"MFR1", int(expected_frame_id), int(expected_timestamp_ns), WIDTH * 2, FRAME_BYTES
    ):
        return None
    return payload[FRAME_SNAPSHOT_HEADER.size:]


def encode_frame_request(frame_id: int, timestamp_ns: int, *, input_fps=None,
                         submitted_fps=None) -> bytes:
    if input_fps is None and submitted_fps is None:
        return REQUEST_HEADER.pack(b"MVR1", int(frame_id), int(timestamp_ns))
    if input_fps is None or submitted_fps is None:
        raise ValueError("Both renderer FPS values must be provided together")
    rates = (float(input_fps), float(submitted_fps))
    if not all(math.isfinite(value) and value >= 0 for value in rates):
        raise ValueError("Renderer FPS values must be finite and non-negative")
    return REQUEST_HEADER_V2.pack(b"MVR2", int(frame_id), int(timestamp_ns), *rates)


def decode_frame_request(packet: bytes):
    if len(packet) == REQUEST_HEADER.size:
        magic, frame_id, timestamp_ns = REQUEST_HEADER.unpack(packet)
        if magic == b"MVR1":
            return frame_id, timestamp_ns
        return None
    if len(packet) == REQUEST_HEADER_V2.size:
        magic, frame_id, timestamp_ns, input_fps, submitted_fps = REQUEST_HEADER_V2.unpack(packet)
        if magic == b"MVR2" and all(math.isfinite(value) and value >= 0
                                     for value in (input_fps, submitted_fps)):
            return frame_id, timestamp_ns, input_fps, submitted_fps
    return None


def encode_ai_result_packet(*, frame_id, timestamp_ns, flags, nav_direction,
                            nav_distance_m, boxes, status=None):
    boxes = list(boxes[:MAX_BOXES])
    if status is not None:
        status_payload = json.dumps(status, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(status_payload) > 32768:
            raise ValueError("AI status payload exceeds the bounded result packet size")
        payload = bytearray(RESULT_HEADER_V2.pack(
            b"MVA2", int(frame_id), int(timestamp_ns), len(boxes), int(flags),
            int(nav_direction), float(nav_distance_m), len(status_payload),
        ))
        for box in boxes:
            if len(box) != 9 or not all(math.isfinite(float(value)) for value in box):
                raise ValueError("Version 2 boxes need seven color/coordinate floats, class ID, and confidence")
            *values, class_id, confidence = box
            if int(class_id) < 0 or int(class_id) > 255 or not 0.0 <= float(confidence) <= 1.0:
                raise ValueError("Detection class ID and confidence are out of range")
            payload.extend(BOX_V2.pack(*(float(value) for value in values), int(class_id), float(confidence)))
        payload.extend(status_payload)
        return bytes(payload)

    payload = bytearray(RESULT_HEADER.pack(
        b"MVA1", int(frame_id), int(timestamp_ns), len(boxes), int(flags),
        int(nav_direction), float(nav_distance_m),
    ))
    for box in boxes:
        if len(box) != 7 or not all(math.isfinite(float(value)) for value in box):
            raise ValueError("AI boxes must contain seven finite float values")
        payload.extend(BOX.pack(*(float(value) for value in box)))
    return bytes(payload)


def decode_ai_result_packet(packet: bytes):
    if len(packet) < RESULT_HEADER.size:
        return None
    magic = packet[:4]
    status = None
    if magic == b"MVA1":
        magic, frame_id, timestamp_ns, count, flags, nav_direction, nav_distance_m = (
            RESULT_HEADER.unpack_from(packet)
        )
        header_bytes, box_struct = RESULT_HEADER.size, BOX
        status_bytes = 0
    elif magic == b"MVA2":
        if len(packet) < RESULT_HEADER_V2.size:
            return None
        magic, frame_id, timestamp_ns, count, flags, nav_direction, nav_distance_m, status_bytes = (
            RESULT_HEADER_V2.unpack_from(packet)
        )
        header_bytes, box_struct = RESULT_HEADER_V2.size, BOX_V2
    else:
        return None
    expected = header_bytes + count * box_struct.size + status_bytes
    if count > MAX_BOXES or expected != len(packet):
        return None
    boxes = [box_struct.unpack_from(packet, header_bytes + index * box_struct.size)
             for index in range(count)]
    if magic == b"MVA2":
        try:
            status = json.loads(packet[header_bytes + count * box_struct.size:].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(status, dict):
            return None
    if not math.isfinite(nav_distance_m):
        return None
    result = {
        "frame_id": frame_id,
        "timestamp_ns": timestamp_ns,
        "flags": flags,
        "nav_direction": nav_direction,
        "nav_distance_m": nav_distance_m,
        "boxes": boxes,
    }
    if status is not None:
        result["status"] = status
    return result


def decode_rgb565_payload(payload: bytes):
    if len(payload) != FRAME_BYTES:
        raise ValueError(f"RGB565 frame must be {FRAME_BYTES} bytes, got {len(payload)}")
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)
    return cv2.cvtColor(packed, cv2.COLOR_BGR5652BGR)


class PackedRgb565:
    """A frame kept in packed form; decode only the quadrants a model needs."""
    __slots__ = ("payload",)

    def __init__(self, payload: bytes):
        if len(payload) != FRAME_BYTES:
            raise ValueError(f"RGB565 frame must be {FRAME_BYTES} bytes, got {len(payload)}")
        self.payload = payload

    def quadrant_bgr(self, quadrant: int):
        if quadrant not in (0, 1, 2, 3):
            raise ValueError("quadrant must be 0, 1, 2, or 3")
        row, col = divmod(quadrant, 2)
        packed = np.frombuffer(self.payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)
        roi = packed[row * HEIGHT // 2:(row + 1) * HEIGHT // 2,
                     col * WIDTH // 2:(col + 1) * WIDTH // 2]
        return cv2.cvtColor(roi, cv2.COLOR_BGR5652BGR)

    def full_bgr(self):
        packed = np.frombuffer(self.payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)
        return cv2.cvtColor(packed, cv2.COLOR_BGR5652BGR)


def select_inference_frames(frame, layout, road_quadrant, navigation_quadrant):
    """Decode the inference and navigation inputs."""
    if isinstance(frame, PackedRgb565):
        if layout == "quad":
            road = frame.quadrant_bgr(road_quadrant)
            navigation = (road if navigation_quadrant == road_quadrant
                          else frame.quadrant_bgr(navigation_quadrant))
            return road, navigation
        road = frame.full_bgr()
        navigation = (road if navigation_quadrant == road_quadrant
                      else frame.quadrant_bgr(navigation_quadrant))
        return road, navigation
    if layout == "quad":
        road = crop_quadrant(frame, road_quadrant)
        navigation = (road if navigation_quadrant == road_quadrant
                      else crop_quadrant(frame, navigation_quadrant))
        return road, navigation
    road = frame
    navigation = (road if navigation_quadrant == road_quadrant
                  else crop_quadrant(frame, navigation_quadrant))
    return road, navigation


def load_replay_frames(paths):
    started = time.perf_counter()
    frames = []
    for path in paths:
        payload = Path(path).read_bytes()
        frames.append(decode_rgb565_payload(payload))
    return frames, (time.perf_counter() - started) * 1000


def crop_quadrant(image, quadrant):
    if quadrant not in (0, 1, 2, 3):
        raise ValueError("quadrant must be 0, 1, 2, or 3")
    height, width = image.shape[:2]
    row, col = divmod(quadrant, 2)
    return image[row * height // 2:(row + 1) * height // 2,
                 col * width // 2:(col + 1) * width // 2]


def _distribution(values, unit="ms"):
    if not values:
        return {"samples": 0, "p50_" + unit: None, "p95_" + unit: None,
                "mean_" + unit: None}
    samples = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(len(samples)),
        "p50_" + unit: float(np.percentile(samples, 50)),
        "p95_" + unit: float(np.percentile(samples, 95)),
        "mean_" + unit: float(np.mean(samples)),
    }


class BoardTelemetry:
    def __init__(self):
        self.samples = {
            name: deque(maxlen=3600)
            for name in ("npu_load_percent", "npu_frequency_mhz",
                         "gpu_load_percent", "gpu_frequency_mhz")
        }
        self.temperatures = deque(maxlen=3600)
        self.running = threading.Event()
        self.worker = None
        self.zones = []
        for index in range(16):
            base = Path(f"/sys/class/thermal/thermal_zone{index}")
            try:
                name = (base / "type").read_text(encoding="ascii").strip()
                (base / "temp").read_text(encoding="ascii")
            except OSError:
                continue
            self.zones.append((name, base / "temp"))

    @staticmethod
    def _number(path):
        try:
            return float(Path(path).read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _load(path):
        try:
            value = Path(path).read_text(encoding="ascii").strip().split("@", 1)[0]
            return float(value)
        except (OSError, ValueError):
            return None

    def start(self):
        self.running.set()
        self.worker = threading.Thread(target=self._sample, name="board-telemetry", daemon=True)
        self.worker.start()

    def _sample(self):
        roots = {
            "npu": "/sys/class/devfreq/fde40000.npu",
            "gpu": "/sys/class/devfreq/fde60000.gpu",
        }
        while self.running.is_set():
            for device, root in roots.items():
                load = self._load(f"{root}/load")
                frequency = self._number(f"{root}/cur_freq")
                if load is not None:
                    self.samples[f"{device}_load_percent"].append(load)
                if frequency is not None:
                    self.samples[f"{device}_frequency_mhz"].append(frequency / 1e6)
            row = {name: [] for name, _ in self.zones}
            for name, path in self.zones:
                value = self._number(path)
                if value is not None:
                    row[name].append(value / 1000.0)
            self.temperatures.append(row)
            time.sleep(0.25)

    def stop(self):
        self.running.clear()
        if self.worker is not None:
            self.worker.join(timeout=1.0)

    def summary(self):
        devices = {
            device: {
                "load_percent": _distribution(
                    self.samples[f"{device}_load_percent"], "percent"
                ),
                "frequency_mhz": _distribution(
                    self.samples[f"{device}_frequency_mhz"], "mhz"
                ),
            }
            for device in ("npu", "gpu")
        }
        zones = {}
        for name, _ in self.zones:
            values = [row[name][0] for row in self.temperatures if row.get(name)]
            if values:
                zones[name] = {
                    "samples": len(values), "min_c": min(values),
                    "max_c": max(values), "end_c": values[-1],
                }
        return {"devices": devices, "temperatures_c": zones,
                "sample_period_ms": 250}


def _current_rss_mb():
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _set_current_python_thread_nice(nice_value=10):
    """Set this Python thread absolutely; never retune RKNN native workers."""
    try:
        tid = threading.get_native_id()
        if os.getpriority(os.PRIO_PROCESS, tid) != int(nice_value):
            os.setpriority(os.PRIO_PROCESS, tid, int(nice_value))
    except (AttributeError, OSError):
        pass


def _deprioritize_sidecar_process(nice_value=10):
    """Preserve measured RKNN caller/native scheduling; lower only the leader."""
    try:
        os.setpriority(os.PRIO_PROCESS, os.getpid(), int(nice_value))
    except (AttributeError, OSError):
        pass


def _quadrant_offset(quadrant):
    row, col = divmod(quadrant, 2)
    return col * (WIDTH // 2), row * (HEIGHT // 2)


def _box_color(name):
    if name == "traffic light":
        return 1.0, 0.68, 0.12
    if name == "person":
        return 1.0, 0.18, 0.18
    return 0.18, 0.92, 0.48


def _navigation_ocr_process(request_queue, result_queue, ttl_ms):
    """Isolated OCR worker: no Python GIL/CPU coupling with YOLO sidecar."""
    try:
        cpus = os.cpu_count() or 4
        if cpus > 1 and os.environ.get("ADAS_OCR_LEGACY_AFFINITY") == "1":
            os.sched_setaffinity(0, {cpus - 1})
        _set_current_python_thread_nice(10)
    except (AttributeError, OSError):
        pass
    cv2.setNumThreads(int(os.environ.get("ADAS_OPENCV_THREADS", "1")))
    reader = Q2NavigationReader(ttl_ms=ttl_ms)
    try:
        reader.warmup()
    except BaseException as exc:
        print(f"NAV_OCR_WARMUP_FAILED {type(exc).__name__}: {exc}", flush=True)
    while True:
        try:
            item = request_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break
        # Never recognize a queued obsolete card while a newer one is available.
        while True:
            try:
                newer=request_queue.get_nowait()
                if newer is None: return
                item=newer
            except queue.Empty: break
        image, frame_id, timestamp_ns = item
        try:
            observation = reader.update(image, frame_id, timestamp_ns)
        except BaseException as exc:
            print(f"NAV_OCR_FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        while True:
            try:
                result_queue.get_nowait()
            except queue.Empty:
                break
        try:
            result_queue.put_nowait(observation.to_dict())
        except queue.Full:
            pass

class AiReplayBridge:
    def __init__(self, args):
        self.args = args
        cv2.setNumThreads(int(os.environ.get("ADAS_AI_OPENCV_THREADS", "4")))
        model_root = Path(args.main_model).resolve().parent
        legacy_root = Path(args.legacy_model_root).resolve() if args.legacy_model_root else None
        if legacy_root is None:
            legacy_root = next(
                (parent for parent in Path(args.main_model).resolve().parents
                 if (parent / "adas-edge-rect").is_dir()),
                Path.home(),
            )
        self.model_profiles = {
            profile.id: profile
            for profile in build_model_profiles(model_root, legacy_root)
        }
        self.model_switch_lock = threading.Lock()
        self._model_options_dirty = True
        self._packets_built = 0
        self.pending_model_switch = None
        self.model_switch_peer = None
        self.model_switch_state = "READY"
        self.model_switch_message = ""
        self.active_model_id = self._profile_id_for_paths(
            args.main_model, args.specialist_model
        ) or "custom"
        self.live_frame_path = Path(args.live_frame_file) if args.live_frame_file else None
        if self.live_frame_path is None:
            self.frames, self.source_prepare_ms = load_replay_frames(args.files)
            if not self.frames:
                raise ValueError("At least one replay frame is required")
        else:
            self.frames, self.source_prepare_ms = [], 0.0
        self.dynamic_hmi_path = Path(args.dynamic_hmi_file) if args.dynamic_hmi_file else None
        self.timing_stream = open(args.timing_log, "a", buffering=1) if args.timing_log else None
        self.last_dynamic_signature = None
        self.dynamic_hmi_updates = 0
        self.latest_dynamic_status = None
        self.last_stopped_refresh_ns = None
        self.shared_frame_mismatches = 0
        self.source_frame_reads = 0
        self.selection_changes = 0
        # Stage samples are intentionally kept in memory for the bounded
        # 60-second board runs. They make the existing timing log actionable
        # without changing the detector cadence or packet contract.
        self.stage_timings = {
            name: deque(maxlen=3600)
            for name in (
                "shared_frame_read", "shared_frame_validate", "rgb565_decode",
                "pipeline", "navigation_submit", "box_mapping", "status_build",
                "dynamic_hmi", "packet_encode", "result_publish",
            )
        }

        active_profile = self.model_profiles.get(self.active_model_id)
        main_size = active_profile.input_size if active_profile else (
            (320, 192) if "320x192" in args.main_model else
            ((416, 256) if "416x256" in args.main_model else (512, 288))
        )
        self.active_backend = active_profile.backend if active_profile else "rknn_lite"
        self.active_task = active_profile.task if active_profile else "detect"
        main = RknnConsoleDetector(
            args.main_model, main_size,
            class_profile=active_profile.class_profile if active_profile else None,
            backend=self.active_backend, task=self.active_task,
        )
        main.single_model = not bool(args.specialist_model)
        try:
            specialist = (
                RknnConsoleDetector(
                    args.specialist_model, (512, 192), specialist=True,
                    class_names=("traffic light",),
                ) if args.specialist_model else None
            )
        except BaseException:
            main.close()
            raise
        self.pipeline = FrontPipeline(
            main, specialist, main_hz=args.main_hz,
            specialist_hz=args.specialist_hz, hazards=False,
            hazards_async=True,
            hazard_hz=(args.hazard_hz if args.hazard_hz > 0 else args.main_hz),
            # Qt already rate-limits live requests to --ai-hz. Running the
            # main model for each latest request avoids a second clock gate
            # dropping requests when PGV timestamps jitter around 100 ms.
            main_every_request=True,
        )
        from inference_mode import ModeControl
        self.modes = ModeControl(self)
        self.event_store = EventStore()
        self.primary_tracks = PrimaryTracks()
        self.light_risk = LightRiskEngine()
        self.lite = False
        self.mosaic_profile_id = "FAST-MOSAIC"
        self.semantics = SemanticWorker()
        self.auto_brake = AutoBrakeController(journal=self.event_store.journal,
            forward_source_id="Q%d" % (self.args.forward_quadrant + 1),
            physical_input=self.args.source=="PGV_LIVE")
        self.navigation = None
        self.nav_request = None
        self.nav_result = None
        self.nav_process = None
        if args.navigation_ocr:
            self.navigation = Q2NavigationReader(ttl_ms=2000)
            # OCR runs in a separate process. Do not load ONNX here: that would
            # contend with RKNN and hold the Python GIL in the YOLO process.
            self.nav_request = multiprocessing.get_context("spawn").Queue(maxsize=1)
            self.nav_result = multiprocessing.get_context("spawn").Queue(maxsize=2)
            self.nav_process = multiprocessing.get_context("spawn").Process(
                target=_navigation_ocr_process,
                args=(self.nav_request, self.nav_result, 2000),
                name="q2-navigation-ocr",
                daemon=True,
            )
            self.nav_process.start()
        self.navigation_last_update_ns = None
        self.navigation_timings = deque(maxlen=3600)
        self.navigation_timing_count = 0
        self.navigation_preprocess_timings = deque(maxlen=3600)
        self.navigation_engine_timings = deque(maxlen=3600)
        self.navigation_jump_rejections = 0
        self.latest_source_timestamp_ns = 0
        self.last_nav_thumb = None
        self.nav_ocr_skipped = 0
        self.navigation_gate = NavigationGate()
        self.navigation_last_force_ns = 0
        self.latest = LatestFrameSlot()
        self.running = threading.Event()
        self.running.set()
        self.request_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.request_socket.bind(("127.0.0.1", args.request_port))
        self.request_socket.settimeout(0.2)
        self.result_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.result_address = ("127.0.0.1", args.result_port)
        self.listener = threading.Thread(target=self._listen, name="mali-ai-requests", daemon=True)
        self.worker = threading.Thread(target=self._infer, name="mali-ai-worker", daemon=True)
        self.request_count = 0
        self.replaced_count = 0
        self.result_count = 0
        self.error_count = 0
        self.latency_ms = deque(maxlen=3600)
        self.worker_cpu_ns = 0
        self.process_cpu_start_ns = time.process_time_ns()
        self.telemetry = BoardTelemetry()
        self.first_request_ns = None
        self.last_request_ns = None
        self.last_error = None
        self.process_start_ns = time.monotonic_ns()
        self.analysis_start_ns = None
        self.debug_enabled = os.environ.get("ADAS_AI_DEBUG", "0") == "1"
        self.debug_log_path = os.environ.get("ADAS_AI_DEBUG_LOG", "")
        self.debug_stream = (
            open(self.debug_log_path, "a", buffering=1)
            if self.debug_log_path else None
        )

    def _debug(self, message):
        if not self.debug_enabled:
            return
        line = f"DEBUG {time.monotonic_ns()} {message}"
        print(line, flush=True)
        if self.debug_stream is not None:
            self.debug_stream.write(line + "\n")

    def start(self):
        status = initial_dynamic_status(self.args.source, time.monotonic_ns())
        status["model_selection"] = self._model_selection_status(include_options=True)
        status["layout"] = self.args.layout
        status["display_view"] = "quad" if self.modes.adapter is not None else "front"
        status["road_quadrant"] = self.args.forward_quadrant
        status["forward_source_id"] = "Q%d" % (self.args.forward_quadrant + 1)
        status["inference"] = self.modes.status()
        self._write_dynamic_hmi(status)
        self.listener.start()
        self.worker.start()
        self._debug(
            f"START listener_alive={self.listener.is_alive()} "
            f"worker_alive={self.worker.is_alive()} hazards={self.args.hazards}"
        )
        _deprioritize_sidecar_process()

    def _listen(self):
        try:
            self._listen_loop()
        except BaseException as exc:
            # A listener failure otherwise leaves the sidecar looking READY
            # while silently dropping every MVR2 request.
            print(f"REQUEST_LISTENER_ERROR {type(exc).__name__}: {exc}", flush=True)

    def _listen_loop(self):
        while self.running.is_set():
            try:
                packet, peer = self.request_socket.recvfrom(128)
            except socket.timeout:
                continue
            except OSError:
                break
            if self.modes.receive(packet, peer):
                continue
            raw_magic = packet[:4]
            packet = self.modes.frame_request(packet)
            if raw_magic == b"MVR3" and self.request_count < 3:
                self._debug(f"FRAME_REQUEST gated={not bool(packet)} mode={self.modes.status()}")
            model_request = decode_model_switch_request(packet)
            if model_request is not None:
                self._queue_model_switch(model_request, peer)
                continue
            # Runtime view selection: MVS1 + road quadrant + navigation
            # quadrant. It is intentionally handled on the request listener so
            # the next inference uses the new crops without restarting RKNN.
            if len(packet) == 6 and packet[:4] == b"MVS1":
                road, navigation = int(packet[4]), int(packet[5])
                if road < 4 and navigation < 4:
                    changed = (road != self.args.road_quadrant or
                               navigation != self.args.navigation_quadrant)
                    self.args.road_quadrant = road
                    self.args.navigation_quadrant = navigation
                    if changed:
                        # Force the navigation reader to run on the next frame
                        # instead of waiting for its old cadence.
                        self.navigation_last_update_ns = None
                        self.selection_changes += 1
                        print(
                            f"SELECT road={road} navigation={navigation}",
                            flush=True,
                        )
                continue
            request = decode_frame_request(packet)
            if request is None:
                continue
            requested_id, requested_timestamp_ns = request[:2]
            input_fps, submitted_fps = request[2:4] if len(request) == 4 else (None, None)
            if self.live_frame_path is not None:
                read_started = time.perf_counter_ns()
                try:
                    snapshot = self.live_frame_path.read_bytes()
                except OSError:
                    self.shared_frame_mismatches += 1
                    continue
                self.stage_timings["shared_frame_read"].append(
                    (time.perf_counter_ns() - read_started) / 1e6
                )
                validate_started = time.perf_counter_ns()
                payload = decode_frame_snapshot(snapshot, requested_id, requested_timestamp_ns)
                self.stage_timings["shared_frame_validate"].append(
                    (time.perf_counter_ns() - validate_started) / 1e6
                )
                if payload is None:
                    self.shared_frame_mismatches += 1
                    if self.shared_frame_mismatches <= 3:
                        self._debug(f"FRAME_IDENTITY_MISMATCH requested={requested_id}/{requested_timestamp_ns} header={snapshot[:28].hex()}")
                    continue
                try:
                    frame = PackedRgb565(payload)
                except ValueError:
                    self.shared_frame_mismatches += 1
                    continue
                now = time.monotonic_ns()
                request = (requested_id, requested_timestamp_ns, frame,
                           input_fps, submitted_fps, now)
                self.source_frame_reads += 1
            else:
                now = time.monotonic_ns()
                request = (requested_id, requested_timestamp_ns, None,
                           input_fps, submitted_fps, now)
            request = (*request, self.modes.token())
            self.request_count += 1
            if self.first_request_ns is None:
                self.first_request_ns = now
            self.last_request_ns = now
            replaced = self.latest.publish(request)
            if replaced:
                self.replaced_count += 1
            if self.debug_enabled and (self.request_count <= 3 or self.request_count % 25 == 0):
                self._debug(
                    f"REQUEST count={self.request_count} frame={requested_id} "
                    f"source_reads={self.source_frame_reads} replaced={replaced} "
                    f"worker_alive={self.worker.is_alive()}"
                )

    def _profile_id_for_paths(self, main_model, specialist_model):
        main_path = Path(main_model).absolute()
        specialist_path = Path(specialist_model).absolute() if specialist_model else None
        for profile_id, profile in self.model_profiles.items():
            if profile.task == "mosaic":
                continue
            if profile.main_model.absolute() != main_path:
                continue
            expected_specialist = (
                profile.specialist_model.absolute()
                if profile.specialist_model is not None else None
            )
            if expected_specialist == specialist_path:
                return profile_id
        return None

    def _stop_navigation_worker(self):
        if self.nav_process is not None:
            try:
                self.nav_request.put_nowait(None)
            except queue.Full:
                pass
            self.nav_process.join(timeout=2.)
            if self.nav_process.is_alive():
                self.nav_process.terminate()
                self.nav_process.join(timeout=1.)
            if self.nav_process.is_alive():
                raise RuntimeError("OCR worker did not stop")
            self.nav_process.close()
        for ipc in (self.nav_request, self.nav_result):
            if ipc is not None:
                ipc.cancel_join_thread()
                ipc.close()
        self.nav_process = self.nav_request = self.nav_result = self.navigation = None

    def _set_lite(self, enabled):
        if enabled == getattr(self,"lite",False):
            return
        if enabled:
            self._stop_navigation_worker()
            if self.semantics is not None:
                self.semantics.close()
                if self.semantics.thread.is_alive():
                    raise RuntimeError("Semantic worker did not stop")
            self.semantics = None
            self._full_control_enabled = self.auto_brake.enabled
            self.auto_brake.enabled = False
            self.auto_brake.pending = None
        else:
            self.semantics = SemanticWorker()
            if self.args.navigation_ocr:
                ctx = multiprocessing.get_context("spawn")
                self.navigation = Q2NavigationReader(ttl_ms=2000)
                self.nav_request = ctx.Queue(maxsize=1)
                self.nav_result = ctx.Queue(maxsize=2)
                self.nav_process = ctx.Process(target=_navigation_ocr_process,
                    args=(self.nav_request,self.nav_result,2000),name="q2-navigation-ocr",daemon=True)
                self.nav_process.start()
            self.auto_brake.enabled = self._full_control_enabled
        self.auto_brake.last_result = None
        self.primary_tracks = PrimaryTracks()
        self.light_risk = LightRiskEngine()
        self.navigation_last_update_ns = None
        self.navigation_gate = NavigationGate()
        self.lite = enabled
        self.latest_dynamic_status = {}
        self._model_options_dirty = True
        print("WORKLOAD " + json.dumps(dict(lite=enabled,ocr_pid=self.nav_process.pid if self.nav_process else None,
            semantics_active=self.semantics is not None,control_enabled=self.auto_brake.enabled,
            control_authorized=self.auto_brake.authorized,timestamp_ns=time.monotonic_ns())),flush=True)

    def _lite_result(self, snapshot, frame_id, timestamp_ns, input_fps, submitted_fps, ai_fps):
        boxes = self._detections_to_boxes(snapshot)
        status = build_dynamic_status(source=self.args.source,frame_id=frame_id,timestamp_ns=timestamp_ns,
            now_ns=time.monotonic_ns(),hud={},navigation={},hazard_timestamp_ns=None,hazards=(),
            detections=list(self.modes.adapter.all_detections),signal_age_ms=None,
            main_age_ms=snapshot.main_age_ms,specialist_age_ms=None,
            input_fps=input_fps,submitted_fps=submitted_fps,ai_fps=ai_fps)
        status.update(model_selection=self._model_selection_status(include_options=True),
            layout=self.args.layout,display_view="quad",road_quadrant=self.args.forward_quadrant,
            inference=self.modes.status(frame_id),box_identities=self.box_identities,event_updates=[],
            semantics={},risk_rules={},control=dict(control_mode=self.auto_brake.mode,enabled=False,
                authorized=self.auto_brake.authorized,inactive_reason="PURE_DETECTION_LITE",context_epoch=self.modes.status()["epoch"]),
            workload=dict(lite=True,ocr_pid=None,semantics_active=False,tesla_active=False,hud_active=False))
        self.latest_dynamic_status = status
        self._last_analysis_cpu_ms = 0.
        return encode_ai_result_packet(frame_id=frame_id,timestamp_ns=timestamp_ns,flags=0,
            nav_direction=0,nav_distance_m=-1.,boxes=boxes,status=status)

    def _model_selection_status(self, include_options=False):
        with self.model_switch_lock:
            active_id = self.active_model_id
            state = self.model_switch_state
            message = self.model_switch_message
            pending = self.pending_model_switch
            requested_id = pending["profile_id"] if pending else ""
        mosaic_active = self.modes.mode == "MOSAIC_FULL"
        active = self.model_profiles.get(active_id)
        status = {
            "active_id": getattr(self,"mosaic_profile_id","FAST-MOSAIC") if mosaic_active else active_id,
            "active_label": "FAST-MOSAIC" if mosaic_active else (active.label if active else "自定义模型"),
            "requested_id": requested_id,
            "state": state,
            "message": message,
        }
        if include_options:
            status["options"] = [profile.wire() for profile in self.model_profiles.values()]
        return status

    def _send_model_switch_reply(self, request_id, peer, state, requested_id, message):
        if peer is None:
            return
        payload = encode_model_switch_reply(
            request_id,
            state=state,
            active_id=getattr(self,"mosaic_profile_id","FAST-MOSAIC") if self.modes.adapter is not None else self.active_model_id,
            requested_id=requested_id,
            message=message,
        )
        if payload is None:
            return
        try:
            self.result_socket.sendto(payload, peer)
            print("MODEL_SWITCH " + json.dumps({"request_id": request_id,
                  "state": state, "requested_id": requested_id,
                  "active_id": getattr(self,"mosaic_profile_id","FAST-MOSAIC") if self.modes.adapter is not None else self.active_model_id,
                  "inference": self.modes.status(), "timestamp_ns": time.monotonic_ns()},
                  separators=(",", ":")), flush=True)
        except OSError:
            pass

    def _queue_model_switch(self, request, peer):
        request_id = int(request["request_id"])
        profile_id = str(request["profile_id"])
        profile = self.model_profiles.get(profile_id)
        if profile is None:
            self._send_model_switch_reply(
                request_id, peer, "REJECTED", profile_id, "未知模型方案"
            )
            return
        if not profile.available:
            self._send_model_switch_reply(
                request_id, peer, "REJECTED", profile_id, profile.unavailable_reason
            )
            return
        with self.model_switch_lock:
            expected_mode = "MOSAIC_FULL" if profile.task == "mosaic" else "ROAD_Q1"
            actual_id = getattr(self,"mosaic_profile_id","FAST-MOSAIC") if self.modes.adapter is not None else self.active_model_id
            if (self.modes.mode == expected_mode and profile_id == actual_id
                    and self.pending_model_switch is None):
                self.model_switch_state = "READY"
                self.model_switch_message = "当前已是该方案"
                immediate_state = "APPLIED"
            else:
                self.pending_model_switch = {
                    "request_id": request_id,
                    "profile_id": profile_id,
                }
                self.model_switch_peer = peer
                self.model_switch_state = "SWITCHING"
                self.model_switch_message = "正在切换模型"
                immediate_state = "SWITCHING"
            message = self.model_switch_message
        self._send_model_switch_reply(
            request_id, peer, immediate_state, profile_id, message
        )

    def _cancel_pending_road_model_switch(self):
        # A newer full-frame choice supersedes a queued road model change.
        with self.model_switch_lock:
            pending, peer = self.pending_model_switch, self.model_switch_peer
            if pending is not None and pending["profile_id"] in ("FAST-MOSAIC", "FAST-MOSAIC-LITE"):
                return
            self.pending_model_switch = self.model_switch_peer = None
            self.model_switch_state = "READY"
            self.model_switch_message = "四路模式使用 FAST-MOSAIC"
        if pending is not None:
            self._send_model_switch_reply(pending["request_id"], peer, "REJECTED",
                                          pending["profile_id"], "已切换四路模式")

    def _apply_pending_model_switch(self):
        with self.model_switch_lock:
            request = self.pending_model_switch
            if request is None:
                return False
            profile = self.model_profiles[request["profile_id"]]
            expected_mode = "MOSAIC_FULL" if profile.task == "mosaic" else "ROAD_Q1"
            if self.modes.mode != expected_mode:
                return False  # HMI owns the epoch; its MVI1 handshake precedes MVM1.
            self.pending_model_switch = None
            peer = self.model_switch_peer
            self.model_switch_peer = None
        request_id = int(request["request_id"])
        profile_id = str(request["profile_id"])
        profile = self.model_profiles[profile_id]
        self._set_lite(profile_id == "FAST-MOSAIC-LITE")
        if profile.task == "mosaic":
            self.mosaic_profile_id = profile_id
            with self.model_switch_lock:
                self.model_switch_state = "APPLIED"
                self.model_switch_message = profile_id + " APPLIED"
            self._send_model_switch_reply(request_id, peer, "APPLIED", profile_id,
                                          self.model_switch_message)
            return True
        old_main = self.pipeline.main_detector
        old_specialist = self.pipeline.specialist_detector
        same_main = (
            Path(self.args.main_model).absolute() == profile.main_model.absolute()
            and self.active_backend == profile.backend
            and self.active_task == profile.task
        )
        current_specialist = (
            Path(self.args.specialist_model).resolve()
            if self.args.specialist_model else None
        )
        wanted_specialist = (
            profile.specialist_model.resolve()
            if profile.specialist_model is not None else None
        )
        same_specialist = current_specialist == wanted_specialist
        new_main = old_main if same_main else None
        new_specialist = old_specialist if same_specialist else None
        try:
            if new_main is None:
                if profile.backend == "fast_mosaic":
                    # ROAD path has already cropped packed RGB565 to physical Q4.
                    # Reuse precisely the same 9-output BDD C ABI as FAST-MOSAIC.
                    from fast_mosaic_backend import FastMosaicDetector
                    new_main = FastMosaicDetector(backend=profile_id)
                else:
                    new_main = RknnConsoleDetector(
                        profile.main_model, profile.input_size,
                        class_profile=profile.class_profile,
                        backend=profile.backend, task=profile.task,
                    )
            if profile.specialist_model is not None and new_specialist is None:
                new_specialist = RknnConsoleDetector(
                    profile.specialist_model, (512, 192), specialist=True,
                    class_names=("traffic light",),
                )
        except Exception as exc:
            if new_main is not None and new_main is not old_main:
                new_main.close()
            if new_specialist is not None and new_specialist is not old_specialist:
                new_specialist.close()
            message = f"模型加载失败：{type(exc).__name__}: {exc}"
            with self.model_switch_lock:
                self.model_switch_state = "FAILED"
                self.model_switch_message = message
            self._send_model_switch_reply(
                request_id, peer, "FAILED", profile_id, message
            )
            return True

        new_main.single_model = profile.specialist_model is None
        self.pipeline.main_detector = new_main
        self.pipeline.specialist_detector = new_specialist
        self.pipeline.reset()
        self.args.main_model = str(profile.main_model)
        self.args.specialist_model = (
            str(profile.specialist_model) if profile.specialist_model else ""
        )
        self.active_backend = profile.backend
        self.active_task = profile.task
        with self.model_switch_lock:
            self.active_model_id = profile_id
            self.model_switch_state = "APPLIED"
            self.model_switch_message = "模型已切换"
        if old_main is not new_main:
            old_main.close()
        if old_specialist is not None and old_specialist is not new_specialist:
            old_specialist.close()
        self._send_model_switch_reply(
            request_id, peer, "APPLIED", profile_id, "模型已切换"
        )
        return True

    def _detections_to_boxes(self, snapshot):
        boxes = []
        self.box_identities = []
        groups = (self.primary_tracks.current,)
        for detections in groups:
            for item in detections or ():
                if len(boxes) >= MAX_BOXES:
                    return boxes
                raw = item.get("bbox")
                if raw is None or len(raw) != 4:
                    continue
                x1, y1, x2, y2 = (float(value) for value in raw)
                if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                    continue
                try:
                    if self.modes.adapter is None:
                        x1, y1, x2, y2 = map_detection_box(
                            (x1, y1, x2, y2), input_layout=self.args.layout,
                            display_view="front", road_quadrant=0,
                        )
                    else:
                        x1, x2 = max(0.0, x1), min(1280.0, x2)
                        y1, y2 = max(0.0, y1), min(720.0, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                except ValueError:
                    continue
                r, g, b = _box_color(item.get("class_name", ""))
                try:
                    confidence = float(item.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                if not math.isfinite(confidence):
                    confidence = 0.0
                self.box_identities.append(dict(track_id=item.get("track_id",0),source_id=item.get("source_id","UNASSIGNED"),
                    velocity=[v*(2 if self.modes.adapter is None else 1) for v in item.get("velocity",[0,0,0,0])]))
                boxes.append((x1, y1, x2, y2, r, g, b,
                              CLASS_IDS.get(str(item.get("class_name", "")), 255),
                              min(1.0, max(0.0, confidence))))
        return boxes

    def _write_dynamic_hmi(self, status):
        if self.dynamic_hmi_path is None:
            return
        signature = json.dumps(status, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if signature == self.last_dynamic_signature:
            return
        pixels = render_dynamic_hmi_rgba(status, self.args.font)
        data = encode_dynamic_hmi_file(status, pixels)
        self.dynamic_hmi_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            prefix=self.dynamic_hmi_path.name + ".", suffix=".tmp",
            dir=str(self.dynamic_hmi_path.parent),
        )
        try:
            with os.fdopen(handle, "wb") as output:
                output.write(data)
            os.replace(temporary, self.dynamic_hmi_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        self.last_dynamic_signature = signature
        self.dynamic_hmi_updates += 1

    def _poll_navigation_ocr(self):
        if self.nav_result is None or self.navigation is None:
            return
        while True:
            try:
                payload = self.nav_result.get_nowait()
            except queue.Empty:
                break
            try:
                observation = NavigationObservation(**payload)
            except (TypeError, ValueError):
                continue
            previous = self.navigation.current(self.latest_source_timestamp_ns)
            if not observation.valid and previous.valid:
                # A single failed OCR recognition must not erase a still-fresh
                # validated card. Keep it until its own TTL expires; the next
                # focused recognition or periodic detection can refresh it.
                self.navigation_timings.append(float(observation.inference_ms))
                self.navigation_timing_count += 1
                continue
            if (observation.valid and previous.valid
                    and previous.distance_m is not None and observation.distance_m is not None):
                if (abs(float(observation.distance_m) - float(previous.distance_m)) > 500.0
                        and observation.ocr_timestamp_ns - previous.ocr_timestamp_ns < 3_000_000_000):
                    self.navigation_jump_rejections += 1
                    continue
            if observation.valid and self.latest_source_timestamp_ns:
                observation = replace(observation, timestamp_ns=self.latest_source_timestamp_ns)
            self.navigation._last = observation
            print(
                f"NAV_OCR_RESULT valid={int(observation.valid)} reason={observation.reason} "
                f"distance={observation.distance_m} road={observation.road_name!r} "
                f"infer_ms={observation.inference_ms:.1f} pre_ms={observation.preprocess_ms:.2f} "
                f"engine_ms={observation.engine_ms:.1f} card_ms={observation.card_locate_ms:.1f} "
                f"arrow_ms={observation.arrow_ms:.1f} dist_ms={observation.distance_ocr_ms:.1f}",
                flush=True,
            )
            self.navigation_timings.append(float(observation.inference_ms))
            self.navigation_timing_count += 1
            self.navigation_preprocess_timings.append(float(observation.preprocess_ms))
            self.navigation_engine_timings.append(float(observation.engine_ms))
    def _analyze(self, frame_id, timestamp_ns, full_frame=None, input_fps=None,
                 submitted_fps=None, ai_fps=None, schedule_timestamp_ns=None):
        if full_frame is None:
            full_frame = self.frames[frame_id % len(self.frames)]
        analysis_cpu_started_ns = time.thread_time_ns()
        decode_cpu_started_ns = time.thread_time_ns()
        decode_start_ns = time.perf_counter_ns()
        road, navigation_image = select_inference_frames(
            full_frame, self.args.layout, self.args.forward_quadrant,
            self.args.navigation_quadrant,
        )
        self._last_decode_ms = (time.perf_counter_ns() - decode_start_ns) / 1e6
        self._last_decode_cpu_ms = (time.thread_time_ns()-decode_cpu_started_ns)/1e6
        self.stage_timings["rgb565_decode"].append(self._last_decode_ms)
        self.latest_source_timestamp_ns = int(timestamp_ns)
        pipeline_cpu_started_ns = time.thread_time_ns()
        pipeline_start_ns = time.perf_counter_ns()
        if self.modes.adapter is not None:
            self.modes.adapter.road_quadrant = self.args.forward_quadrant
            self.modes.adapter.full_frame = (full_frame.full_bgr() if isinstance(full_frame, PackedRgb565) else full_frame)
        self.pipeline.advance(
            road, int(frame_id), int(timestamp_ns),
            schedule_timestamp_ns=schedule_timestamp_ns,
        )
        self._last_pipeline_ms = (time.perf_counter_ns() - pipeline_start_ns) / 1e6
        self._last_pipeline_cpu_ms = (time.thread_time_ns()-pipeline_cpu_started_ns)/1e6
        self.stage_timings["pipeline"].append(self._last_pipeline_ms)
        navigation_submit_started = time.perf_counter_ns()
        self._poll_navigation_ocr()
        if (self.navigation is not None and
                (self.navigation_last_update_ns is None or
                 timestamp_ns - self.navigation_last_update_ns >= round(1e9 / self.args.ocr_hz))):
            current_nav=self.navigation.current(int(timestamp_ns))
            changed=self.navigation_gate.wants(navigation_image,int(timestamp_ns),
                current_nav.valid,current_nav.ocr_timestamp_ns)
            self.navigation_last_update_ns=int(timestamp_ns)
            if changed:
                nav960=cv2.resize(navigation_image,(960,540),interpolation=cv2.INTER_LINEAR)
                job=(nav960,int(frame_id),int(timestamp_ns))
                try:
                    self.nav_request.put_nowait(job)
                    self.navigation_gate.submitted(navigation_image,int(timestamp_ns))
                except queue.Full:
                    # Replace, rather than waiting behind, a stale pending request.
                    try: self.nav_request.get_nowait()
                    except queue.Empty: pass
                    try:
                        self.nav_request.put_nowait(job)
                        self.navigation_gate.submitted(navigation_image,int(timestamp_ns))
                    except queue.Full: self.nav_ocr_skipped+=1
            else: self.nav_ocr_skipped+=1
        self.stage_timings["navigation_submit"].append(
            (time.perf_counter_ns() - navigation_submit_started) / 1e6
        )
        lane = next(iter(Lane))
        snapshot = self.pipeline.snapshot(lane, int(timestamp_ns))
        mode = self.modes.status()
        detections=(list(snapshot.main)+list(snapshot.specialist) if self.modes.adapter is None else self.modes.adapter.all_detections)
        tracking_cpu_start=time.thread_time_ns()
        tracking_start=time.perf_counter_ns()
        self.primary_tracks.update(detections,(mode["mode"],mode["epoch"],self.active_model_id,self.args.forward_quadrant),timestamp_ns)
        self._last_tracking_ms=(time.perf_counter_ns()-tracking_start)/1e6
        self._last_tracking_cpu_ms=(time.thread_time_ns()-tracking_cpu_start)/1e6
        if self.lite:
            return self._lite_result(snapshot,frame_id,timestamp_ns,input_fps,submitted_fps,ai_fps)
        self.stage_timings.setdefault("risk_fast",deque(maxlen=3600))
        risk_start=time.perf_counter_ns()
        risk_context=(mode["mode"],mode["epoch"],self.active_model_id,self.args.forward_quadrant)
        detector=getattr(self.pipeline.main_detector,"detector",None)
        masks=getattr(detector,"last_masks",None) if self.modes.adapter is None else None
        letterbox=getattr(detector,"last_letterbox",None) if masks is not None else None
        self.semantics.submit(road,self.primary_tracks.current,timestamp_ns,risk_context,masks,letterbox)
        semantic_result=self.semantics.current(timestamp_ns,risk_context)
        candidates=self.light_risk.update(self.primary_tracks.current,timestamp_ns,
            risk_context,semantic_result) if self.args.hazards else []
        events = self.event_store.update(candidates, mode=mode["mode"], epoch=mode["epoch"],
            bridge_id=mode["bridge_id"], model_id=self.active_model_id, frame_id=frame_id,
            timestamp_ns=timestamp_ns, measurement_ns=timestamp_ns, road_quadrant=self.args.forward_quadrant)
        self.stage_timings["risk_fast"].append((time.perf_counter_ns()-risk_start)/1e6)
        snapshot=replace(snapshot,hazards=RiskObservation(timestamp_ns=timestamp_ns,
            events=events,tracks=tuple(self.primary_tracks.current),frame_id=frame_id,
            source_id="Q%d" % (self.args.forward_quadrant + 1)))
        self.auto_brake.update(snapshot.hazards,context=mode)
        nav = self.navigation.current(int(timestamp_ns)).to_dict() if self.navigation else {}
        hud = build_hud_perception(road, snapshot)
        flags = 0
        if hud.traffic == TrafficCode.RED:
            flags |= FLAG_RED
        elif hud.traffic == TrafficCode.GREEN:
            flags |= FLAG_GREEN
        if snapshot.hazards.pedestrian_hazard:
            flags |= FLAG_PEDESTRIAN_HAZARD
        if snapshot.hazards.obstacle_hazard:
            flags |= FLAG_OBSTACLE_HAZARD
        if nav.get("valid"):
            flags |= FLAG_NAVIGATION_VALID
        nav_direction = NAV_DIRECTIONS.get(nav.get("direction", "none"), 0)
        nav_distance = nav.get("distance_m")
        box_started = time.perf_counter_ns()
        boxes = self._detections_to_boxes(snapshot)
        self.stage_timings["box_mapping"].append(
            (time.perf_counter_ns() - box_started) / 1e6
        )
        ages = [value for value in (snapshot.main_age_ms, snapshot.specialist_age_ms)
                if isinstance(value, (int, float))]
        status_started = time.perf_counter_ns()
        status = build_dynamic_status(
            source=self.args.source,
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            now_ns=time.monotonic_ns(),
            hud=hud.to_dict(),
            navigation=nav,
            canonical_risks=True,
            hazard_timestamp_ns=snapshot.hazards.timestamp_ns,
            hazards=snapshot.hazards.events,
            detections=list(snapshot.main) + list(snapshot.specialist),
            signal_age_ms=min(ages) if ages else None,
            main_age_ms=snapshot.main_age_ms,
            specialist_age_ms=snapshot.specialist_age_ms,
            navigation_ttl_ms=2000,
            input_fps=input_fps,
            submitted_fps=submitted_fps,
            ai_fps=ai_fps,
        )
        self._packets_built += 1
        include_options = self._model_options_dirty or (self._packets_built % 50 == 0)
        status["model_selection"] = self._model_selection_status(
            include_options=include_options
        )
        status["layout"] = self.args.layout
        status["display_view"] = "quad" if self.modes.adapter is not None else "front"
        status["road_quadrant"] = self.args.forward_quadrant
        status["forward_source_id"] = "Q%d" % (self.args.forward_quadrant + 1)
        status["inference"] = self.modes.status(frame_id)
        status["workload"] = dict(lite=False,ocr_pid=self.nav_process.pid if self.nav_process else None,
            semantics_active=self.semantics.thread.is_alive(),ocr_alive=self.nav_process.is_alive() if self.nav_process else False,
            tesla_active=True,hud_active=True)
        status["semantics"] = dict(semantic_result,metrics=self.semantics.metrics())
        status["risk_rules"] = self.light_risk.metrics()
        status["box_identities"] = self.box_identities
        status["event_updates"] = [{k:e[k] for k in ("event_id", "lifecycle_state", "cleared_ns") if k in e}
                                   for e in self.event_store.updates if e["lifecycle_state"] == "CLEARED"]
        status["control"] = dict(self.auto_brake.last_result or {}, control_mode=self.auto_brake.mode,
            enabled=self.auto_brake.enabled, authorized=self.auto_brake.authorized,
            forward_source_id=self.auto_brake.forward_source_id,context_epoch=mode["epoch"])
        road = getattr(snapshot.hazards, "road", None)
        if road:
            status["road_scene"] = road
        self.stage_timings["status_build"].append(
            (time.perf_counter_ns() - status_started) / 1e6
        )
        self.latest_dynamic_status = status
        hmi_started = time.perf_counter_ns()
        self._write_dynamic_hmi(status)
        self.stage_timings["dynamic_hmi"].append(
            (time.perf_counter_ns() - hmi_started) / 1e6
        )
        packet_started = time.perf_counter_ns()
        try:
            packet = encode_ai_result_packet(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                flags=flags,
                nav_direction=nav_direction,
                nav_distance_m=float(nav_distance) if nav_distance is not None else -1.0,
                boxes=boxes,
                status=status,
            )
            self._model_options_dirty = False
        except ValueError:
            # Never let an oversized option list break inference: retry with
            # the compact per-frame status and keep the menu from the last
            # packet that fit.
            status["model_selection"].pop("options", None)
            packet = encode_ai_result_packet(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                flags=flags,
                nav_direction=nav_direction,
                nav_distance_m=float(nav_distance) if nav_distance is not None else -1.0,
                boxes=boxes,
                status=status,
            )
        self.stage_timings["packet_encode"].append(
            (time.perf_counter_ns() - packet_started) / 1e6
        )
        self._last_analysis_cpu_ms=(time.thread_time_ns()-analysis_cpu_started_ns)/1e6
        return packet

    def _infer(self):
        cpu_profile = None
        if os.environ.get("ADAS_CPU_PROFILE"):
            import cProfile
            cpu_profile=cProfile.Profile(timer=time.thread_time)
            cpu_profile.enable()
        completed_times = deque()
        rate_token = None
        while self.running.is_set():
            self.modes.apply()
            request = self.latest.take(timeout=0.2)
            if request is None:
                self._apply_pending_model_switch()
                continue
            if self.debug_enabled and (self.result_count < 3 or self.result_count % 25 == 0):
                self._debug(
                    f"WORKER_TAKE frame={request[0]} result_count={self.result_count}"
                )
            self._apply_pending_model_switch()
            if len(request) == 7:
                *request, mode_token = request
                if mode_token != self.modes.token():
                    continue
            if len(request) == 6:
                frame_id, timestamp_ns, full_frame, input_fps, submitted_fps, ai_recv_ns = request
            else:
                frame_id, timestamp_ns, full_frame, input_fps, submitted_fps = request
                ai_recv_ns = time.monotonic_ns()
            key = (frame_id, timestamp_ns, self.modes.epoch)
            if key == self.modes.last_result_key:
                continue
            infer_start_ns = time.monotonic_ns()
            started = time.perf_counter_ns()
            cpu_started = time.thread_time_ns()
            try:
                now_ns = time.monotonic_ns()
                if self.analysis_start_ns is None:
                    self.analysis_start_ns = now_ns
                token = (self.modes.token(), self.active_model_id)
                if token != rate_token:
                    completed_times.clear()
                    rate_token = token
                while completed_times and now_ns - completed_times[0] > 1_200_000_000:
                    completed_times.popleft()
                span = (completed_times[-1] - completed_times[0]) / 1e9 if len(completed_times) > 1 else 0
                ai_fps = (len(completed_times) - 1) / span if span else 0.0
                main_count_before = self.pipeline.main_update_count
                specialist_count_before = self.pipeline.specialist_update_count
                packet = self._analyze(
                    frame_id, timestamp_ns, full_frame, input_fps, submitted_fps,
                    ai_fps, schedule_timestamp_ns=ai_recv_ns,
                )
                if self.debug_enabled and (self.result_count < 3 or self.result_count % 25 == 0):
                    self._debug(
                        f"ANALYZE_DONE frame={frame_id} bytes={len(packet)} "
                        f"main_samples={len(self.pipeline.main_timings)}"
                    )
                # Keep the measured RKNN inference-thread scheduling. OCR uses
                # an absolute nice 10; only the bridge leader is deprioritized.
                _deprioritize_sidecar_process()
                infer_end_ns = time.monotonic_ns()
                publish_started = time.perf_counter_ns()
                self.result_socket.sendto(packet, self.result_address)
                publish_ns = time.monotonic_ns()
                completed_times.append(publish_ns)
                self.modes.last_result_key = key
                self.stage_timings["result_publish"].append(
                    (time.perf_counter_ns() - publish_started) / 1e6
                )
                if self.timing_stream is not None:
                    main_timing = getattr(self.pipeline.main_detector.detector, "last_timing", None)
                    specialist_runtime = getattr(self.pipeline.specialist_detector, "detector", None)
                    specialist_timing = getattr(specialist_runtime, "last_timing", None)
                    self.timing_stream.write(json.dumps({
                        "frame_id": int(frame_id),
                        "inference": self.modes.status(frame_id),
                        "model_id": getattr(self,"mosaic_profile_id","FAST-MOSAIC") if self.modes.adapter else self.active_model_id,
                        "ai_fps": ai_fps,
                        "capture_timestamp_ns": int(timestamp_ns),
                        "ai_receive_timestamp_ns": int(ai_recv_ns),
                        "infer_start_ns": int(infer_start_ns),
                        "infer_end_ns": int(infer_end_ns),
                        "schedule_timestamp_ns": int(ai_recv_ns),
                        "decode_ms": getattr(self, "_last_decode_ms", None),
                        "shared_frame_read_ms": self.stage_timings["shared_frame_read"][-1] if self.stage_timings["shared_frame_read"] else None,
                        "shared_frame_validate_ms": self.stage_timings["shared_frame_validate"][-1] if self.stage_timings["shared_frame_validate"] else None,
                        "pipeline_ms": getattr(self, "_last_pipeline_ms", None),
                        "navigation_submit_ms": self.stage_timings["navigation_submit"][-1] if self.stage_timings["navigation_submit"] else None,
                        "box_mapping_ms": self.stage_timings["box_mapping"][-1] if self.stage_timings["box_mapping"] else None,
                        "status_build_ms": self.stage_timings["status_build"][-1] if self.stage_timings["status_build"] else None,
                        "dynamic_hmi_ms": self.stage_timings["dynamic_hmi"][-1] if self.stage_timings["dynamic_hmi"] else None,
                        "packet_encode_ms": self.stage_timings["packet_encode"][-1] if self.stage_timings["packet_encode"] else None,
                        "result_publish_ms": self.stage_timings["result_publish"][-1] if self.stage_timings["result_publish"] else None,
                        "main_ran": self.pipeline.main_update_count > main_count_before,
                        "specialist_ran": self.pipeline.specialist_update_count > specialist_count_before,
                        "main_count": self.pipeline.main_update_count,
                        "specialist_count": self.pipeline.specialist_update_count,
                        "main_ms": (self.pipeline.main_timings[-1] if self.pipeline.main_timings else None),
                        "main_timing": main_timing if self.pipeline.main_update_count > main_count_before else None,
                        "specialist_timing": specialist_timing if self.pipeline.specialist_update_count > specialist_count_before else None,
                        "result_publish_timestamp_ns": int(publish_ns),
                        "boxes": int(struct.unpack_from("<I", packet, 20)[0]),
                        "traffic": self.latest_dynamic_status.get("traffic"),
                        "traffic_reason": self.latest_dynamic_status.get("traffic_reason"),
                        "detection_counts": timing_detection_counts(self.latest_dynamic_status),
                        "specialist_age_ms": self.latest_dynamic_status.get("specialist_age_ms"),
                        "navigation_state": (self.latest_dynamic_status.get("navigation") or {}).get("state"),
                        "navigation_ocr_age_ms": (self.latest_dynamic_status.get("navigation") or {}).get("ocr_age_ms"),
                        "hazards": self.latest_dynamic_status.get("hazards"),
                        "hazard_details": self.latest_dynamic_status.get("hazard_details"),
                        "event_updates": self.latest_dynamic_status.get("event_updates"),
                        "analysis_thread_cpu_ms":self._last_analysis_cpu_ms,
                        "decode_thread_cpu_ms":self._last_decode_cpu_ms,
                        "pipeline_thread_cpu_ms":self._last_pipeline_cpu_ms,
                        "tracking_thread_cpu_ms":self._last_tracking_cpu_ms,
                        "tracking_ms":self._last_tracking_ms,
                        "risk_fast_ms":self.stage_timings["risk_fast"][-1],
                        "risk_rules":self.latest_dynamic_status.get("risk_rules"),
                        "semantics":self.latest_dynamic_status.get("semantics"),
                        "control":self.latest_dynamic_status.get("control"),
                        "workload":self.latest_dynamic_status.get("workload"),
                        "forward_source_id":self.latest_dynamic_status.get("forward_source_id"),
                    }, separators=(",", ":")) + "\n")
                self.result_count += 1
                self.latency_ms.append((time.perf_counter_ns() - started) / 1e6)
                if cpu_profile is not None and self.result_count % 100 == 0:
                    cpu_profile.dump_stats(os.environ["ADAS_CPU_PROFILE"])
                    cpu_profile.enable()
            except BaseException as exc:
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._debug(
                    f"INFER_ERROR frame={request[0]} {self.last_error}\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                self.worker_cpu_ns += time.thread_time_ns() - cpu_started

    def run(self):
        self.telemetry.start()
        self.start()
        print(
            f"READY evidence={self.args.source} request_port={self.args.request_port} "
            f"result_port={self.args.result_port} frames={len(self.frames)} "
            f"shared_frame_file={self.live_frame_path or 'disabled'}",
            flush=True,
        )
        try:
            while self.running.is_set():
                time.sleep(0.1)
                now_ns = time.monotonic_ns()
                idle_origin_ns = self.last_request_ns or self.process_start_ns
                if should_refresh_stopped_status(
                        now_ns, idle_origin_ns, self.last_stopped_refresh_ns):
                    self._publish_stopped_status(now_ns)
                if (self.args.source != "PGV_LIVE" and self.args.idle_timeout > 0 and
                        now_ns - idle_origin_ns > int(self.args.idle_timeout * 1e9)):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        self.auto_brake.close()
        if self.semantics is not None:self.semantics.close()
        self.event_store.journal.close()
        self.running.clear()
        self.latest.close()
        try:
            self.request_socket.close()
        except OSError:
            pass
        self.listener.join(timeout=1.0)
        self.worker.join(timeout=10.0)
        self._publish_stopped_status()
        self.result_socket.close()
        self._stop_navigation_worker()
        if self.timing_stream is not None:
            self.timing_stream.close()
            self.timing_stream = None
        if self.debug_stream is not None:
            self.debug_stream.close()
            self.debug_stream = None
        self.telemetry.stop()
        self.modes.shutdown()
        self.pipeline.main_detector.close()
        if self.pipeline.specialist_detector is not None:
            self.pipeline.specialist_detector.close()
        self._write_summary()

    def _publish_stopped_status(self, now_ns=None):
        if self.dynamic_hmi_path is None:
            return
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if (self.last_stopped_refresh_ns is not None and
                now_ns - self.last_stopped_refresh_ns < DYNAMIC_HMI_HEARTBEAT_NS):
            return
        previous = self.latest_dynamic_status or initial_dynamic_status(
            self.args.source, now_ns
        )
        navigation = previous.get("navigation") or {}
        stopped = build_dynamic_status(
            source=self.args.source,
            frame_id=int(previous.get("frame_id", 0)),
            timestamp_ns=now_ns,
            now_ns=now_ns,
            hud={"traffic": "OFF", "traffic_evidence": "SOURCE_STOPPED"},
            navigation={
                "source_frame_id": navigation.get("source_frame_id"),
                "timestamp_ns": None,
            },
            hazard_timestamp_ns=None,
            hazards=(),
            detections=(),
            signal_age_ms=None,
            main_age_ms=None,
            specialist_age_ms=None,
            input_fps=previous.get("input_fps"),
            submitted_fps=previous.get("submitted_fps"),
            ai_fps=previous.get("ai_fps"),
            source_stopped=True,
            source_available=False,
        )
        self.latest_dynamic_status = stopped
        self._write_dynamic_hmi(stopped)
        self.last_stopped_refresh_ns = now_ns

    def _write_summary(self):
        duration_ns = 0
        if self.first_request_ns is not None and self.last_request_ns is not None:
            duration_ns = self.last_request_ns - self.first_request_ns
        timings = {
            "main_rknn": _distribution(self.pipeline.main_timings),
            "specialist_rknn": _distribution(self.pipeline.specialist_timings),
            "navigation_ocr": _distribution(self.navigation_timings),
            "hazard_postprocess": _distribution(
                self.pipeline.hazard_engine.timings
                if self.pipeline.hazard_engine is not None else []
            ),
            "all_ai_work": _distribution(self.latency_ms),
        }
        for name, values in self.stage_timings.items():
            timings[name] = _distribution(values)
        hazard_breakdowns = getattr(self.pipeline.hazard_engine, "breakdowns", []) \
            if self.pipeline.hazard_engine is not None else []
        main_detector = getattr(self.pipeline.main_detector, "detector", None)
        specialist_detector = getattr(self.pipeline.specialist_detector, "detector", None)
        for label, detector in (("main", main_detector), ("specialist", specialist_detector)):
            history = list(getattr(detector, "timing_history", ()) or ())
            for key in ("preprocess_ms", "rknn_call_ms", "decode_nms_ms", "total_ms"):
                timings[f"{label}_{key}"] = _distribution(
                    [float(row[key]) for row in history if key in row]
                )
        peak_rss_mb = None
        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            peak_rss_mb = float(usage.ru_maxrss) / 1024.0
        duration_s = duration_ns / 1e9 if duration_ns else 0.0
        process_cpu_ns = time.process_time_ns() - self.process_cpu_start_ns
        online_cpus = os.cpu_count() or 1
        rknpu_updates = getattr(self.pipeline, "main_update_count", len(self.pipeline.main_timings))
        specialist_updates = getattr(
            self.pipeline, "specialist_update_count", len(self.pipeline.specialist_timings)
        )
        ocr_updates = self.navigation_timing_count
        report = {
            "evidence": self.args.source,
            "layout": self.args.layout,
            "display_view": self.args.display_view,
            "frame_format": "1280x720 RGB565_LE packed, decoded using the existing PGV BGR565 conversion",
            "model_paths": {
                "yolo11n": str(self.args.main_model),
                "traffic_light_specialist": str(self.args.specialist_model),
            },
            "source_frames_loaded": len(self.frames),
            "source_prepare_ms": self.source_prepare_ms,
            "shared_frame_file_reads": self.source_frame_reads,
            "shared_frame_identity_mismatches": self.shared_frame_mismatches,
            "input_requests_received": self.request_count,
            "input_request_fps": self.request_count / duration_s if duration_s else 0.0,
            "source_ids_analyzed": self.result_count,
            "source_id_analysis_fps": self.result_count / duration_s if duration_s else 0.0,
            "source_ids_replaced_before_analysis": self.replaced_count,
            "result_packets_sent": self.result_count,
            "dynamic_hmi_texture_updates": self.dynamic_hmi_updates,
            "inference_errors": self.error_count,
            "last_error": self.last_error,
            "updates_per_second": {
                "yolo11n_rknn": rknpu_updates / duration_s if duration_s else 0.0,
                "traffic_light_specialist_rknn": specialist_updates / duration_s if duration_s else 0.0,
                "navigation_ocr": ocr_updates / duration_s if duration_s else 0.0,
            },
            "timings_ms": timings,
            "hazard_breakdown": {
                name: _distribution([
                    float(row[name]) for row in hazard_breakdowns
                    if name in row and row[name] is not None
                ])
                for name in ("track_ms", "road_ms", "motion_ms", "brake_ms", "fusion_ms")
            },
            "road_scene_enabled": bool(getattr(getattr(self.pipeline, "hazard_engine", None), "road_enabled", True)),
            "resources": {
                "process_cpu_one_core_percent": 100.0 * process_cpu_ns / duration_ns if duration_ns else 0.0,
                "process_cpu_machine_percent": (
                    100.0 * process_cpu_ns / duration_ns / online_cpus if duration_ns else 0.0
                ),
                "inference_worker_cpu_one_core_percent": (
                    100.0 * self.worker_cpu_ns / duration_ns if duration_ns else 0.0
                ),
                "online_cpus": online_cpus,
                "rss_end_mb": _current_rss_mb(),
                "rss_peak_mb": peak_rss_mb,
                "board_telemetry": self.telemetry.summary(),
            },
            "display_note": "AI runs in a separate latest-only worker and never blocks EGL frame submission.",
        }
        path = self.args.summary
        if path:
            Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


def _read_npu_load_samples():
    samples = []
    for path in ("/sys/kernel/debug/rknpu/load", "/sys/class/devfreq/fde40000.npu/load"):
        try:
            value = Path(path).read_text(encoding="ascii").strip()
        except OSError:
            continue
        samples.append({"path": path, "value": value})
    return samples


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--file", dest="files", action="append", default=[])
    result.add_argument("--live-frame-file", default="")
    result.add_argument("--dynamic-hmi-file", default="")
    result.add_argument("--font", default="")
    result.add_argument("--source", choices=("SYNTHETIC", "LOCAL_REPLAY", "PGV_LIVE"),
                        default="LOCAL_REPLAY")
    result.add_argument("--layout", choices=("quad", "front", "full"), default="quad")
    result.add_argument("--display-view", choices=("quad", "front"), default="quad")
    default_model_root = Path.home() / "cityscapes_arch_compare_20261027" / "models"
    result.add_argument(
        "--main-model",
        default=str(default_main_model_path(default_model_root)),
        help=f"main RKNN model (default profile: {DEFAULT_MODEL_ID})",
    )
    result.add_argument("--specialist-model", default="")
    result.add_argument("--legacy-model-root", default="")
    result.add_argument("--request-port", type=int, default=38260)
    result.add_argument("--result-port", type=int, default=38261)
    result.add_argument("--forward-quadrant", type=int, choices=range(4), default=3)
    result.add_argument("--road-quadrant", type=int, choices=range(4), default=0)
    result.add_argument("--navigation-quadrant", type=int, choices=range(4), default=1)
    result.add_argument("--main-hz", type=float, default=10.0)
    result.add_argument(
        "--hazard-hz", type=float, default=0.0,
        help="hazard analysis cadence; 0 follows the YOLO cadence",
    )
    result.add_argument("--specialist-hz", type=float, default=3.0)
    result.add_argument(
        "--hazards", dest="hazards", action="store_true", default=True,
        help="enable hazard event analysis (default)",
    )
    result.add_argument(
        "--no-hazards", dest="hazards", action="store_false",
        help="disable hazard event analysis while keeping YOLO inference",
    )
    result.add_argument("--ocr-hz", type=float, default=3.0)
    result.add_argument(
        "--navigation-ocr", dest="navigation_ocr", action="store_true", default=False,
        help="enable the low-priority Q2 navigation-card OCR worker (default: off)",
    )
    result.add_argument(
        "--no-navigation-ocr", dest="navigation_ocr", action="store_false",
        help="disable navigation OCR (performance isolation/debug)",
    )
    result.add_argument("--idle-timeout", type=float, default=2.0)
    result.add_argument("--summary", default="")
    result.add_argument("--timing-log", default="", help="JSONL timing records for frame/AI latency analysis")
    return result


def main():
    args = parser().parse_args()
    if bool(args.live_frame_file) == bool(args.files):
        raise SystemExit("Specify exactly one of --live-frame-file or one or more --file options")
    if args.layout == "front" and args.display_view == "quad":
        raise SystemExit("A front-only input cannot be shown as a four-camera view")
    if args.layout == "full" and args.display_view != "quad":
        raise SystemExit("Full-frame inference must be displayed as the four-camera view")
    bridge = AiReplayBridge(args)

    def _terminate(signum, _frame):
        try:
            bridge.close()
        finally:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    bridge.run()


if __name__ == "__main__":
    main()
