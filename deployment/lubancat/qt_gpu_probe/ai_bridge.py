#!/usr/bin/env python3
"""Latest-only RKNN/OCR sidecar for the Mali RGB565 renderer."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import multiprocessing
import os
import queue
import signal
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
from q2_navigation_reader import CARD_ROI, Q2NavigationReader
from tools.competition_console.navigation_reader import NavigationObservation
from tools.competition_console.navigation_reader import NavigationObservation
from tools.competition_console.pipeline import FrontPipeline
from tools.competition_console.rknn_detector import RknnConsoleDetector
from tools.competition_console.latest_frame_slot import LatestFrameSlot


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
    if input_layout not in ("quad", "front") or display_view not in ("quad", "front"):
        raise ValueError("input_layout and display_view must be quad or front")
    if road_quadrant not in (0, 1, 2, 3):
        raise ValueError("road_quadrant must be 0, 1, 2, or 3")
    values = tuple(float(value) for value in bbox)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("bbox must contain four finite values")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")
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
                         source_stopped=False, source_available=True,
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
            and hazard_age_ms < DYNAMIC_RESULT_TTL_MS):
        for item in hazards:
            if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
                continue
            kind = item["kind"]
            hazard_kinds.append(kind)
            hazard_details.append({
                "event_type": kind,
                "source_id": source,
                "source_frame_id": int(frame_id),
                "timestamp": int(hazard_timestamp_ns),
                "severity": "warning",
                "valid": True,
                "ttl_ms": DYNAMIC_RESULT_TTL_MS,
                "evidence": item.get("source") or "IMAGE_PERCEPTION",
            })
    if (not source_stopped and source_available and traffic == "RED"
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
        if len(status_payload) > 4096:
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
        self.samples = {"npu_load_percent": [], "npu_frequency_mhz": [],
                        "gpu_load_percent": [], "gpu_frequency_mhz": []}
        self.temperatures = []
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
        if cpus > 1:
            os.sched_setaffinity(0, {cpus - 1})
        os.nice(10)
    except (AttributeError, OSError):
        pass
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

        main_size = (320, 192) if "320x192" in args.main_model else ((416, 256) if "416x256" in args.main_model else (512, 288))
        main = RknnConsoleDetector(args.main_model, main_size)
        try:
            specialist = RknnConsoleDetector(
                args.specialist_model, (512, 192), specialist=True
            )
        except BaseException:
            main.close()
            raise
        self.pipeline = FrontPipeline(
            main, specialist, main_hz=args.main_hz,
            specialist_hz=args.specialist_hz, hazards=True,
        )
        self.navigation = None
        self.nav_request = None
        self.nav_result = None
        self.nav_process = None
        if args.navigation_ocr:
            self.navigation = Q2NavigationReader(ttl_ms=12000)
            # OCR runs in a separate process. Do not load ONNX here: that would
            # contend with RKNN and hold the Python GIL in the YOLO process.
            self.nav_request = multiprocessing.get_context("spawn").Queue(maxsize=1)
            self.nav_result = multiprocessing.get_context("spawn").Queue(maxsize=2)
            self.nav_process = multiprocessing.get_context("spawn").Process(
                target=_navigation_ocr_process,
                args=(self.nav_request, self.nav_result, 12000),
                name="q2-navigation-ocr",
                daemon=True,
            )
            self.nav_process.start()
        self.navigation_last_update_ns = None
        self.navigation_timings = []
        self.navigation_preprocess_timings = []
        self.navigation_engine_timings = []
        self.navigation_jump_rejections = 0
        self.latest_source_timestamp_ns = 0
        self.last_nav_thumb = None
        self.nav_ocr_skipped = 0
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
        self.frame_ids = []
        self.request_times = []
        self.latency_ms = []
        self.worker_cpu_ns = 0
        self.process_cpu_start_ns = time.process_time_ns()
        self.telemetry = BoardTelemetry()
        self.first_request_ns = None
        self.last_request_ns = None
        self.last_error = None
        self.process_start_ns = time.monotonic_ns()
        self.analysis_start_ns = None

    def start(self):
        self._write_dynamic_hmi(initial_dynamic_status(self.args.source, time.monotonic_ns()))
        self.listener.start()
        self.worker.start()

    def _listen(self):
        while self.running.is_set():
            try:
                packet, _ = self.request_socket.recvfrom(128)
            except socket.timeout:
                continue
            except OSError:
                break
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
                try:
                    snapshot = self.live_frame_path.read_bytes()
                except OSError:
                    self.shared_frame_mismatches += 1
                    continue
                payload = decode_frame_snapshot(snapshot, requested_id, requested_timestamp_ns)
                if payload is None:
                    self.shared_frame_mismatches += 1
                    continue
                try:
                    frame = decode_rgb565_payload(payload)
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
            self.request_count += 1
            self.request_times.append(now)
            if self.first_request_ns is None:
                self.first_request_ns = now
            self.last_request_ns = now
            if self.latest.publish(request):
                self.replaced_count += 1

    def _detections_to_boxes(self, snapshot):
        boxes = []
        for detections in (snapshot.main, snapshot.specialist):
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
                    x1, y1, x2, y2 = map_detection_box(
                        (x1, y1, x2, y2), input_layout=self.args.layout,
                        display_view=self.args.display_view,
                        road_quadrant=self.args.road_quadrant,
                    )
                except ValueError:
                    continue
                r, g, b = _box_color(item.get("class_name", ""))
                try:
                    confidence = float(item.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                if not math.isfinite(confidence):
                    confidence = 0.0
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
                f"engine_ms={observation.engine_ms:.1f}",
                flush=True,
            )
            self.navigation_timings.append(float(observation.inference_ms))
            self.navigation_preprocess_timings.append(float(observation.preprocess_ms))
            self.navigation_engine_timings.append(float(observation.engine_ms))
    def _analyze(self, frame_id, timestamp_ns, full_frame=None, input_fps=None,
                 submitted_fps=None, ai_fps=None):
        if full_frame is None:
            full_frame = self.frames[frame_id % len(self.frames)]
        if self.args.layout == "quad":
            road = crop_quadrant(full_frame, self.args.road_quadrant)
            navigation_image = crop_quadrant(full_frame, self.args.navigation_quadrant)
        else:
            road = full_frame
            navigation_image = full_frame
        self.latest_source_timestamp_ns = int(timestamp_ns)
        self.pipeline.advance(road, int(frame_id), int(timestamp_ns))
        self._poll_navigation_ocr()
        if (self.navigation is not None and
                (self.navigation_last_update_ns is None or
                 timestamp_ns - self.navigation_last_update_ns >= round(1e9 / self.args.ocr_hz))):
            # Skip OCR when the navigation screen is visually unchanged and the
            # last result is still fresh. This avoids burning CPU on a static map.
            x0, y0, x1, y1 = CARD_ROI
            card = navigation_image[y0:y1, x0:x1]
            thumb = cv2.resize(cv2.cvtColor(card, cv2.COLOR_BGR2GRAY),
                               (96, 45), interpolation=cv2.INTER_AREA)
            current_nav = self.navigation.current(int(timestamp_ns))
            changed = True
            if self.last_nav_thumb is not None and current_nav.valid:
                diff = np.abs(thumb.astype(np.int16) - self.last_nav_thumb.astype(np.int16)).mean()
                changed = diff > 3.0
            # Force a periodic full re-check even if the card looks static.
            force = timestamp_ns - self.navigation_last_force_ns >= 10_000_000_000
            if force:
                changed = True
                self.navigation_last_force_ns = int(timestamp_ns)
            self.last_nav_thumb = thumb
            self.navigation_last_update_ns = int(timestamp_ns)
            if changed:
                nav960 = cv2.resize(navigation_image, (960, 540), interpolation=cv2.INTER_LINEAR)
                try:
                    self.nav_request.put_nowait((nav960, int(frame_id), int(timestamp_ns)))
                except queue.Full:
                    self.nav_ocr_skipped += 1
            else:
                self.nav_ocr_skipped += 1
        lane = next(iter(Lane))
        snapshot = self.pipeline.snapshot(lane, int(timestamp_ns))
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
        boxes = self._detections_to_boxes(snapshot)
        ages = [value for value in (snapshot.main_age_ms, snapshot.specialist_age_ms)
                if isinstance(value, (int, float))]
        status = build_dynamic_status(
            source=self.args.source,
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            now_ns=time.monotonic_ns(),
            hud=hud.to_dict(),
            navigation=nav,
            hazard_timestamp_ns=snapshot.hazards.timestamp_ns,
            hazards=snapshot.hazards.events,
            detections=list(snapshot.main) + list(snapshot.specialist),
            signal_age_ms=min(ages) if ages else None,
            main_age_ms=snapshot.main_age_ms,
            specialist_age_ms=snapshot.specialist_age_ms,
            navigation_ttl_ms=12000,
            input_fps=input_fps,
            submitted_fps=submitted_fps,
            ai_fps=ai_fps,
        )
        self.latest_dynamic_status = status
        self._write_dynamic_hmi(status)
        return encode_ai_result_packet(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            flags=flags,
            nav_direction=nav_direction,
            nav_distance_m=float(nav_distance) if nav_distance is not None else -1.0,
            boxes=boxes,
            status=status,
        )

    def _infer(self):
        while self.running.is_set():
            request = self.latest.take(timeout=0.2)
            if request is None:
                continue
            if len(request) == 6:
                frame_id, timestamp_ns, full_frame, input_fps, submitted_fps, ai_recv_ns = request
            else:
                frame_id, timestamp_ns, full_frame, input_fps, submitted_fps = request
                ai_recv_ns = time.monotonic_ns()
            infer_start_ns = time.monotonic_ns()
            started = time.perf_counter_ns()
            cpu_started = time.thread_time_ns()
            try:
                now_ns = time.monotonic_ns()
                if self.analysis_start_ns is None:
                    self.analysis_start_ns = now_ns
                analysis_duration_s = max((now_ns - self.analysis_start_ns) / 1e9, 1e-6)
                ai_fps = (self.result_count + 1) / analysis_duration_s
                packet = self._analyze(
                    frame_id, timestamp_ns, full_frame, input_fps, submitted_fps, ai_fps
                )
                infer_end_ns = time.monotonic_ns()
                self.result_socket.sendto(packet, self.result_address)
                publish_ns = time.monotonic_ns()
                if self.timing_stream is not None:
                    self.timing_stream.write(json.dumps({
                        "frame_id": int(frame_id),
                        "capture_timestamp_ns": int(timestamp_ns),
                        "ai_receive_timestamp_ns": int(ai_recv_ns),
                        "infer_start_ns": int(infer_start_ns),
                        "infer_end_ns": int(infer_end_ns),
                        "result_publish_timestamp_ns": int(publish_ns),
                        "boxes": int(struct.unpack_from("<I", packet, 20)[0]),
                        "traffic": self.latest_dynamic_status.get("traffic"),
                        "traffic_reason": self.latest_dynamic_status.get("traffic_reason"),
                        "specialist_age_ms": self.latest_dynamic_status.get("specialist_age_ms"),
                        "navigation_state": (self.latest_dynamic_status.get("navigation") or {}).get("state"),
                        "navigation_ocr_age_ms": (self.latest_dynamic_status.get("navigation") or {}).get("ocr_age_ms"),
                        "hazards": self.latest_dynamic_status.get("hazards"),
                        "hazard_details": self.latest_dynamic_status.get("hazard_details"),
                    }, separators=(",", ":")) + "\n")
                self.result_count += 1
                self.frame_ids.append(frame_id)
                self.latency_ms.append((time.perf_counter_ns() - started) / 1e6)
            except BaseException as exc:
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
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
                if (self.args.source != "PGV_LIVE" and
                        now_ns - idle_origin_ns > int(self.args.idle_timeout * 1e9)):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
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
        if self.nav_process is not None:
            try:
                self.nav_request.put_nowait(None)
            except queue.Full:
                pass
            self.nav_process.join(timeout=2.0)
            if self.nav_process.is_alive():
                self.nav_process.terminate()
                self.nav_process.join(timeout=1.0)
            self.nav_process = None
        if self.timing_stream is not None:
            self.timing_stream.close()
            self.timing_stream = None
        self.telemetry.stop()
        self.pipeline.main_detector.close()
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
        peak_rss_mb = None
        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            peak_rss_mb = float(usage.ru_maxrss) / 1024.0
        duration_s = duration_ns / 1e9 if duration_ns else 0.0
        process_cpu_ns = time.process_time_ns() - self.process_cpu_start_ns
        online_cpus = os.cpu_count() or 1
        rknpu_updates = len(self.pipeline.main_timings)
        specialist_updates = len(self.pipeline.specialist_timings)
        ocr_updates = len(self.navigation_timings)
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
            "source_ids_analyzed": len(self.frame_ids),
            "source_id_analysis_fps": len(self.frame_ids) / duration_s if duration_s else 0.0,
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
    result.add_argument("--layout", choices=("quad", "front"), default="quad")
    result.add_argument("--display-view", choices=("quad", "front"), default="quad")
    result.add_argument("--main-model", required=True)
    result.add_argument("--specialist-model", required=True)
    result.add_argument("--request-port", type=int, default=38260)
    result.add_argument("--result-port", type=int, default=38261)
    result.add_argument("--road-quadrant", type=int, choices=range(4), default=3)
    result.add_argument("--navigation-quadrant", type=int, choices=range(4), default=1)
    result.add_argument("--main-hz", type=float, default=10.0)
    result.add_argument("--specialist-hz", type=float, default=3.0)
    result.add_argument("--ocr-hz", type=float, default=3.0)
    result.add_argument(
        "--navigation-ocr", dest="navigation_ocr", action="store_true", default=True,
        help="enable the low-priority Q2 navigation-card OCR worker (default: on)",
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
