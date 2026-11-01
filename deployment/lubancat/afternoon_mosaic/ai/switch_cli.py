import socket
import sys
import time

sys.path.insert(0, "/home/cat/qt_gpu_probe_ai_20261019")
from model_profiles import decode_model_switch_reply, encode_model_switch_request

profile = sys.argv[1]
request_id = int(sys.argv[2]) if len(sys.argv) > 2 else int(time.time()) & 0xFFFF
packet = encode_model_switch_request(request_id, profile)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 0))
sock.settimeout(6.0)
started = time.perf_counter()
sock.sendto(packet, ("127.0.0.1", 38260))
while True:
    data, _ = sock.recvfrom(4096)
    reply = decode_model_switch_reply(data)
    if reply and int(reply.get("request_id", 0)) == request_id:
        print(reply, "elapsed_ms", round((time.perf_counter() - started) * 1000, 1), flush=True)
        if reply.get("state") in ("APPLIED", "FAILED", "REJECTED"):
            break
sock.close()
