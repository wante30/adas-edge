import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path("/home/cat/competition_cockpit_port_20261018")
TEST_DIR = Path("/home/cat/b05n_p2_capi_integration_20261031")
PROD_DIR = Path("/home/cat/qt_gpu_probe_ai_20261019")
os.environ["ADAS_EDGE_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROD_DIR))
sys.path.insert(0, str(TEST_DIR))

import capi_detector
capi_detector.LIB_PATH = str(TEST_DIR / "libp2capi.so.new")

from model_profiles import decode_model_switch_reply, encode_model_switch_request

spec = importlib.util.spec_from_file_location("ai_mtl_test", TEST_DIR / "ai_bridge.py")
bridge_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge_module
spec.loader.exec_module(bridge_module)
comp_spec = importlib.util.spec_from_file_location("comp_mtl_test", TEST_DIR / "b05p_competition_bridge.py")
comp_module = importlib.util.module_from_spec(comp_spec)
sys.modules[comp_spec.name] = comp_module
comp_spec.loader.exec_module(comp_module)
comp_module._install_raw_head_adapter(bridge_module, "B")

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
               5: "bus", 7: "truck", 9: "traffic light"}
replay = Path("/home/cat/b05n_mtl_b2a_bench_20261031/replay60")
files = sorted(replay.glob("frame_*.rgb565"))
req_port, res_port = 39621, 39622
argv = [
    "--source", "SYNTHETIC", "--layout", "quad", "--display-view", "front",
    "--road-quadrant", "3", "--navigation-quadrant", "1",
    "--main-model", "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn",
    "--main-hz", "10", "--hazard-hz", "5", "--specialist-hz", "1",
    "--request-port", str(req_port), "--result-port", str(res_port),
    "--idle-timeout", "120",
]
for path in files:
    argv.extend(("--file", str(path)))
args = bridge_module.parser().parse_args(argv)
bridge = bridge_module.AiReplayBridge(args)
bridge.start()
time.sleep(0.5)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 0))
sock.settimeout(8.0)
res_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
res_sock.bind(("127.0.0.1", res_port))
res_sock.settimeout(3.0)
state = {"request": 5000}


def switch(profile_id):
    state["request"] += 1
    request_id = state["request"]
    packet = encode_model_switch_request(request_id, profile_id)
    started = time.perf_counter()
    sock.sendto(packet, ("127.0.0.1", req_port))
    while True:
        data, _ = sock.recvfrom(4096)
        reply = decode_model_switch_reply(data)
        if reply and int(reply.get("request_id", 0)) == request_id:
            if reply.get("state") in ("APPLIED", "FAILED", "REJECTED"):
                return reply, (time.perf_counter() - started) * 1000


def frames(n=3):
    for _ in range(n):
        state["request"] += 1
        sock.sendto(bridge_module.encode_frame_request(state["request"], time.monotonic_ns()),
                    ("127.0.0.1", req_port))
        time.sleep(0.08)
    got = []
    deadline = time.time() + 2.5
    while len(got) < n and time.time() < deadline:
        try:
            data, _ = res_sock.recvfrom(65535)
        except socket.timeout:
            break
        result = bridge_module.decode_ai_result_packet(data)
        if result:
            classes = [CLASS_NAMES.get(int(box[7]), int(box[7])) for box in result["boxes"]]
            status = result.get("status", {})
            selection = status.get("model_selection", {})
            size = len(json.dumps(status, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            got.append((result["frame_id"], classes, "options" in selection, size))
    return got


def detector_info():
    main = bridge.pipeline.main_detector
    capi = getattr(main, "capi", None)
    return {
        "type": type(main).__name__,
        "backend": getattr(main, "backend", None),
        "task": getattr(main, "task", None),
        "capi": type(capi).__name__ if capi else None,
        "mask": (getattr(capi, "mask_w", None), getattr(capi, "mask_h", None)),
        "overlay": os.path.exists(capi_detector.OVERLAY_PATH),
    }


def report(label):
    info = detector_info()
    print(label, "model=", bridge.active_model_id, info, flush=True)
    for frame in frames():
        print("  result", frame[0], frame[1], "options=", frame[2], "status_bytes=", frame[3], flush=True)
    info = detector_info()
    meta = None
    try:
        with open("/dev/shm/adas_mtl_overlay.json") as stream:
            meta = json.load(stream)
    except OSError:
        pass
    print("  after", info, "meta", meta, flush=True)


try:
    print("options", [(o["id"], o["available"]) for o in bridge._model_selection_status(include_options=True)["options"][:7]], flush=True)
    report("B")
    for target in ("B-capi", "MTL-FULL", "MTL-FAST", "B", "MTL-FAST", "B-capi", "B"):
        reply, ms = switch(target)
        print("switch", target, reply["state"], round(ms, 1), flush=True)
        report(target)
finally:
    bridge.close()
    sock.close()
    res_sock.close()
print("done", flush=True)
