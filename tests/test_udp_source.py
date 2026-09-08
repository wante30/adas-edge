import socket

import cv2
import numpy as np

from adas_edge.video.udp import UDPVideoSource


def test_udp_source_decodes_complete_jpeg_datagram() -> None:
    source = UDPVideoSource("127.0.0.1", 0, camera_id="fpga-front")
    port = source._socket.getsockname()[1]
    ok, jpeg = cv2.imencode(".jpg", np.full((6, 8, 3), 50, dtype=np.uint8))
    assert ok
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(jpeg.tobytes(), ("127.0.0.1", port))
        frame = source.read()
    finally:
        sender.close()
        source.close()

    assert frame is not None
    assert frame.camera_id == "fpga-front"
    assert frame.width == 8
    assert frame.height == 6
