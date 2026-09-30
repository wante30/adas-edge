"""FastAPI/WebSocket transport for SceneFrame and optional front-camera JPEGs."""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .navigation_state import NavigationState
from .scene_builder import SceneBuilder
from .schemas import NavigationIntent
from .video_pipeline import FramePacket


@dataclass
class SimulatorOptions:
    source: Any
    builder: SceneBuilder
    navigation: NavigationState
    frontend_dir: Path
    source_name: str
    target_hz: float = 30.0
    video_jpeg_quality: int = 78


def _jpeg_data(image: Any, quality: int) -> str | None:
    if image is None:
        return None
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def create_app(options: SimulatorOptions) -> FastAPI:
    app = FastAPI(title="ADAS HMI PC Simulator", docs_url=None, redoc_url=None)
    started_at = time.perf_counter()
    counts = {"frames": 0, "last_frame_ms": 0, "last_source_s": 0.0, "last_ws_s": 0.0, "detector_updates": 0}

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True, "mode": options.source_name, "frames": counts["frames"]})

    @app.get("/api/config")
    async def config() -> JSONResponse:
        return JSONResponse({"mode": options.source_name, "target_hz": options.target_hz, "protocol": "SceneFrame/v1", "distance_mode": "PSEUDO_DISTANCE"})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        last_tick = time.perf_counter()
        last_frame_time = None
        try:
            while True:
                tick_started = time.perf_counter()
                packet: FramePacket | None = await asyncio.to_thread(options.source.next)
                if packet is None:
                    await websocket.send_json({"type": "end", "message": "input complete"})
                    break
                route_nav = options.navigation.at(packet.frame_id, packet.timestamp_ms)
                metadata = packet.metadata or {}
                # Replay rows may use a nested lane contract or the flat mock contract.
                lane_metadata = metadata.get("lanes") or metadata.get("lane")
                scene = options.builder.build(
                    packet.frame_id,
                    packet.timestamp_ms,
                    packet.detections,
                    int(packet.image.shape[1]) if packet.image is not None else 960,
                    int(packet.image.shape[0]) if packet.image is not None else 540,
                    traffic_light_state=str(metadata.get("traffic_light_state", metadata.get("traffic_light", "UNKNOWN"))),
                    lane_metadata=lane_metadata,
                    navigation=route_nav,
                    warning_flags=list(metadata.get("warning_flags", [])),
                    detector_update=packet.detector_update,
                    source=options.source_name,
                )
                # Preserve the transport and device-stage evidence in the
                # renderer-facing frame. The browser can inspect these fields
                # without knowing the source implementation.
                for key in ("system_mode", "channel_roles", "pcie", "rk3566"):
                    if key in metadata:
                        scene.metadata[key] = metadata[key]
                now = time.perf_counter()
                frame_delta = now - last_tick
                last_tick = now
                counts["frames"] += 1
                counts["last_frame_ms"] = packet.timestamp_ms
                counts["last_source_s"] = frame_delta
                counts["last_ws_s"] = now - tick_started
                if packet.detector_update:
                    counts["detector_updates"] += 1
                wall = max(0.001, now - started_at)
                scene.metrics.update({
                    "backend_ws_rate": round(counts["frames"] / wall, 2),
                    "video_fps": round(1.0 / frame_delta, 2) if frame_delta > 0 else 0,
                    "detector_fps": round(counts["detector_updates"] / wall, 2),
                    "render_target_fps": options.target_hz,
                })
                if packet.stream_images:
                    scene.metrics["stream_count"] = len(packet.stream_images)
                    if packet.transport:
                        scene.metrics["pcie_bytes_total"] = int(packet.transport.get("pcie_bytes", 0))
                message = {"type": "scene", "scene": scene.to_dict(), "video_jpeg": _jpeg_data(packet.image, options.video_jpeg_quality)}
                if packet.stream_images:
                    # ``video_jpeg`` already carries the front channel; send
                    # only the three additional thumbnails to avoid doubling
                    # the largest JPEG on every WebSocket tick.
                    message["video_jpegs"] = {role: _jpeg_data(image, options.video_jpeg_quality) for role, image in packet.stream_images.items() if role != "front"}
                if packet.transport:
                    message["transport"] = dict(packet.transport)
                await websocket.send_json(message)
                wait = 1.0 / options.target_hz - (time.perf_counter() - tick_started)
                if wait > 0:
                    await asyncio.sleep(wait)
                last_frame_time = now
        except WebSocketDisconnect:
            return
        finally:
            # Sources own their capture/model handles; they are closed after a client leaves.
            if hasattr(options.source, "close"):
                await asyncio.to_thread(options.source.close)

    app.mount("/", StaticFiles(directory=str(options.frontend_dir), html=True), name="frontend")
    return app
