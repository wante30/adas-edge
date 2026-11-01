import importlib.util
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path("/home/cat/competition_cockpit_port_20261018")
BRIDGE_DIR = Path("/home/cat/qt_gpu_probe_ai_20261019")
os.environ["ADAS_EDGE_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BRIDGE_DIR))

from model_profiles import decode_model_switch_reply, encode_model_switch_request

spec = importlib.util.spec_from_file_location("ai_bridge_test", BRIDGE_DIR / "ai_bridge.py")
bridge_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge_module
spec.loader.exec_module(bridge_module)
comp_spec = importlib.util.spec_from_file_location("b05p_comp_test", BRIDGE_DIR / "b05p_competition_bridge.py")
comp_module = importlib.util.module_from_spec(comp_spec)
sys.modules[comp_spec.name] = comp_module
comp_spec.loader.exec_module(comp_module)
comp_module._install_raw_head_adapter(bridge_module, "B")

replay = Path("/home/cat/b05n_mtl_b2a_bench_20261031/replay60")
files = sorted(replay.glob("frame_*.rgb565"))
req_port = 39551
res_port = 39552
argv = [
    "--source", "SYNTHETIC", "--layout", "quad", "--display-view", "front",
    "--road-quadrant", "3", "--navigation-quadrant", "1",
    "--main-model", "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn",
    "--main-hz", "10", "--hazard-hz", "5", "--specialist-hz", "1",
    "--request-port", str(req_port), "--result-port", str(res_port),
    "--idle-timeout", "60",
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
res_sock.settimeout(2.0)


def switch(profile_id, request_id):
    packet = encode_model_switch_request(request_id, profile_id)
    started = time.perf_counter()
    sock.sendto(packet, ("127.0.0.1", req_port))
    while True:
        data, _ = sock.recvfrom(4096)
        reply = decode_model_switch_reply(data)
        if reply and int(reply.get("request_id", 0)) == request_id:
            if reply.get("state") in ("APPLIED", "FAILED", "REJECTED"):
                return reply, (time.perf_counter() - started) * 1000


def frames(n=5):
    for i in range(n):
        sock.sendto(bridge_module.encode_frame_request(9000 + i, time.monotonic_ns()),
                    ("127.0.0.1", req_port))
        time.sleep(0.05)
    got = []
    deadline = time.time() + 2.0
    while len(got) < n and time.time() < deadline:
        try:
            data, _ = res_sock.recvfrom(65535)
        except socket.timeout:
            break
        result = bridge_module.decode_ai_result_packet(data)
        if result:
            got.append((result["frame_id"], len(result["boxes"])))
    return got


try:
    options = bridge._model_selection_status()["options"] if hasattr(bridge, "_model_state") else []
except Exception:
    options = []
print("options", [(o.get("id"), o.get("available")) for o in options][:8], flush=True)
print("has B-capi option", any(o.get("id") == "B-capi" for o in options), flush=True)
print("start", bridge.active_model_id, type(bridge.pipeline.main_detector).__name__, flush=True)
print("frames B", frames(), flush=True)
for cycle in range(3):
    r, ms = switch("B-capi", 100 + cycle * 2)
    print("switch B-capi", r, round(ms, 2), bridge.active_model_id,
          type(bridge.pipeline.main_detector).__name__,
          getattr(bridge.pipeline.main_detector, "capi", None) is not None, flush=True)
    print("frames capi", frames(), flush=True)
    r, ms = switch("B", 101 + cycle * 2)
    print("switch B", r, round(ms, 2), bridge.active_model_id,
          type(bridge.pipeline.main_detector).__name__, flush=True)
    print("frames B", frames(), flush=True)
bridge.close()
sock.close()
print("done", flush=True)
