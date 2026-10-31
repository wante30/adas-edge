"""RK3566 model choices exposed by the Qt live inference sidecar."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct


MODEL_SWITCH_REQUEST_MAGIC = b"MVM1"
MODEL_SWITCH_REPLY_MAGIC = b"MMS1"
DEFAULT_MODEL_ID = "B"
B05N_P2_FILENAME = "b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn"
_MODEL_ID = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
_REQUEST_HEADER = struct.Struct("<IB")
_REQUEST_ID = struct.Struct("<I")


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    main_model: Path
    specialist_model: Path | None
    input_size: tuple[int, int]
    class_profile: str
    legacy: bool = False
    backend: str = "rknn_lite"
    task: str = "detect"

    @property
    def available(self) -> bool:
        return self.main_model.is_file() and (
            self.specialist_model is None or self.specialist_model.is_file()
        )

    @property
    def unavailable_reason(self) -> str:
        missing = [path for path in (self.main_model, self.specialist_model)
                   if path is not None and not path.is_file()]
        return "模型文件未安装" if missing else ""

    def wire(self) -> dict[str, object]:
        """Compact option record used by the bounded MVA2 status payload."""
        return {
            "id": self.id,
            "label": self.label,
            "available": self.available,
            "legacy": self.legacy,
            "has_specialist": self.specialist_model is not None,
            "reason": self.unavailable_reason,
        }

    def public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "available": self.available,
            "legacy": self.legacy,
            "backend": self.backend,
            "task": self.task,
            "has_specialist": self.specialist_model is not None,
            "reason": self.unavailable_reason,
        }


def build_model_profiles(model_root: str | Path,
                         legacy_root: str | Path) -> tuple[ModelProfile, ...]:
    model_root = Path(model_root)
    legacy_root = Path(legacy_root)
    bdd = model_root / "bdd_yolov8n_512x288_raw_heads_int8.rknn"
    p2 = default_main_model_path(model_root)
    capi = model_root / "b05n_p2_multiclass_capi.rknn"
    mtl_full = Path("/home/cat/models/b05n_p2_mtl_b2a/b05n_p2_mtl_b2a_int8.rknn")
    mtl_fast = Path("/home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn")
    specialist = model_root / "b05l_v8n_p2_raw_heads_int8_500_20261026.rknn"
    old_rect_models = legacy_root / "adas-edge-rect" / "rect_models"
    old_tl_models = legacy_root / "adas-edge-rect" / "tl_check"
    old_bdd_kl = old_rect_models / "bdd_yolov8n_512x288_int8_kl.rknn"
    old_bdd = legacy_root / "b03e_quant_20261022" / "bdd_yolov8n_512x288_int8.rknn"
    old_yolo_320 = old_rect_models / "yolo11n_rk3566_int8_split_320x192.rknn"
    old_yolo_416 = old_rect_models / "yolo11n_rk3566_int8_split_416x256.rknn"
    old_yolo_512 = old_rect_models / "yolo11n_rk3566_int8_split_512x288.rknn"
    old_yolo_512_mmse = old_rect_models / "yolo11n_rk3566_int8_split_512x288_mmse.rknn"
    old_yolo_640 = old_rect_models / "yolo11n_rk3566_int8_split_640x352.rknn"
    old_specialist = old_tl_models / "tl_specialist_512x192_hybrid2.rknn"

    profiles = [
        ModelProfile(DEFAULT_MODEL_ID, "B · B05N P2 十类 Raw-head INT8", p2, None, (512, 288), "bdd10"),
        ModelProfile("B-capi", "B-CAPI · B05N P2 Raw-head INT8 (C API)", capi, None, (512, 288), "bdd10", backend="rknn_capi"),
        ModelProfile("MTL-FULL", "B-2a · Full Resolution MTL (C API)", mtl_full, None, (512, 288), "bdd10", backend="rknn_capi", task="mtl"),
        ModelProfile("MTL-FAST", "B-2a · Seg-256 MTL (C API)", mtl_fast, None, (512, 288), "bdd10", backend="rknn_capi", task="mtl"),
        ModelProfile("A", "A · BDD-YOLOv8n Raw-head INT8", bdd, None, (512, 288), "bdd10"),
        ModelProfile("C", "C · BDD + B05L Specialist", bdd, specialist, (512, 288), "bdd10"),
        ModelProfile("D", "D · B05N P2 + B05L Specialist", p2, specialist, (512, 288), "bdd10"),
        ModelProfile("legacy-bdd-kl", "旧版 · BDD-YOLOv8n KL INT8", old_bdd_kl, None,
                     (512, 288), "bdd10", legacy=True),
        ModelProfile("legacy-bdd-int8", "旧版 · BDD-YOLOv8n INT8", old_bdd, None,
                     (512, 288), "bdd10", legacy=True),
        ModelProfile("legacy-yolo11n-320", "旧版 · YOLO11n INT8 320×192",
                     old_yolo_320, None, (320, 192), "coco80", legacy=True),
        ModelProfile("legacy-yolo11n-416", "旧版 · YOLO11n INT8 416×256",
                     old_yolo_416, None, (416, 256), "coco80", legacy=True),
        ModelProfile("legacy-yolo11n", "旧版 · YOLO11n INT8 512×288",
                     old_yolo_512, None, (512, 288), "coco80", legacy=True),
        ModelProfile("legacy-yolo11n-mmse", "旧版 · YOLO11n MMSE 512×288",
                     old_yolo_512_mmse, None, (512, 288), "coco80", legacy=True),
        ModelProfile("legacy-yolo11n-640", "旧版 · YOLO11n INT8 640×352",
                     old_yolo_640, None, (640, 352), "coco80", legacy=True),
        ModelProfile("legacy-yolo11n-specialist", "旧版 · YOLO11n 512×288 + Specialist",
                     old_yolo_512, old_specialist, (512, 288), "coco80", legacy=True),
        ModelProfile("legacy-bdd-kl-specialist", "旧版 · BDD KL + Specialist",
                     old_bdd_kl, old_specialist, (512, 288), "bdd10", legacy=True),
    ]

    old_specialist_versions = (
        ("hybrid2", "tl_specialist_512x192_hybrid2.rknn"),
        ("logits", "tl_specialist_512x192_logits.rknn"),
        ("hybrid", "tl_specialist_512x192_hybrid.rknn"),
        ("int8-split", "tl_specialist_rk3566_int8_split_512x192.rknn"),
        ("traincalib", "tl_specialist_512x192_traincalib.rknn"),
    )
    for profile_id, filename in old_specialist_versions:
        specialist_path = old_tl_models / filename
        profiles.append(ModelProfile(
            f"legacy-bdd-specialist-{profile_id}",
            f"旧专模 · BDD + Specialist {profile_id}",
            bdd, specialist_path, (512, 288), "bdd10", legacy=True,
        ))

    return tuple(profiles)


def default_main_model_path(model_root: str | Path) -> Path:
    return Path(model_root) / B05N_P2_FILENAME


def encode_model_switch_request(request_id: int, profile_id: str) -> bytes | None:
    if not 1 <= int(request_id) <= 0xFFFFFFFF:
        return None
    if not isinstance(profile_id, str) or not _MODEL_ID.fullmatch(profile_id):
        return None
    encoded_id = profile_id.encode("ascii")
    return (MODEL_SWITCH_REQUEST_MAGIC
            + _REQUEST_HEADER.pack(int(request_id), len(encoded_id))
            + encoded_id)


def decode_model_switch_request(packet: bytes) -> dict[str, object] | None:
    if not isinstance(packet, bytes) or len(packet) < 10 or len(packet) > 57:
        return None
    if packet[:4] != MODEL_SWITCH_REQUEST_MAGIC:
        return None
    try:
        request_id, id_bytes = _REQUEST_HEADER.unpack_from(packet, 4)
        if len(packet) != 9 + id_bytes:
            return None
        profile_id = packet[9:].decode("ascii")
    except (UnicodeDecodeError, struct.error):
        return None
    if request_id == 0 or not _MODEL_ID.fullmatch(profile_id):
        return None
    return {"request_id": request_id, "profile_id": profile_id}


def encode_model_switch_reply(request_id: int, *, state: str, active_id: str,
                              requested_id: str, message: str = "") -> bytes | None:
    if not 1 <= int(request_id) <= 0xFFFFFFFF:
        return None
    if not all(isinstance(value, str) and _MODEL_ID.fullmatch(value)
               for value in (active_id, requested_id)):
        return None
    payload = json.dumps({
        "state": str(state),
        "active_id": active_id,
        "requested_id": requested_id,
        "message": str(message),
    }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return MODEL_SWITCH_REPLY_MAGIC + _REQUEST_ID.pack(int(request_id)) + payload


def decode_model_switch_reply(packet: bytes) -> dict[str, object] | None:
    if not isinstance(packet, bytes) or len(packet) < 10 or len(packet) > 2048:
        return None
    if packet[:4] != MODEL_SWITCH_REPLY_MAGIC:
        return None
    try:
        request_id = _REQUEST_ID.unpack_from(packet, 4)[0]
        fields = json.loads(packet[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, struct.error):
        return None
    if request_id == 0 or not isinstance(fields, dict):
        return None
    if not all(isinstance(fields.get(key), str)
               for key in ("state", "active_id", "requested_id", "message")):
        return None
    return {"request_id": request_id, **fields}
