"""HMI-owned inference epochs; legacy quadrant selection never changes mode."""
import json
import struct
import threading
import time
import uuid

ROAD = "ROAD_Q1"
MOSAIC = "MOSAIC_FULL"
SPACES = {ROAD: "ROAD_VIEW_1280x720", MOSAIC: "MOSAIC_1280x720"}
COMMAND = struct.Struct("<4sQI")
REQUEST = struct.Struct("<4sQQQff")


class MosaicRoadAdapter:
    """One full-frame inference, with only unambiguous Q1 objects sent to safety."""
    def __init__(self, detector, owns_detector=False):
        self.owns_detector = owns_detector
        self.full_detector = detector
        self.detector = self
        self.all_detections = []
        self.full_frame = None
        self.last_timing = None
        self.single_model = True

    def detect(self, road):
        assert self.full_frame is not None
        self.all_detections, elapsed = self.full_detector.detect(self.full_frame)
        self.last_timing = getattr(getattr(self.full_detector, "detector", self.full_detector), "last_timing", None)
        q1 = []
        for item in self.all_detections:
            x1, y1, x2, y2 = item["bbox"]
            # Crossing boxes remain visible, but never become a road command.
            if 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 360:
                q1.append(dict(item))
        return q1, elapsed

    def close(self):
        if self.owns_detector:
            self.full_detector.close()
            self.owns_detector = False


class ModeControl:
    def __init__(self, bridge, mosaic_factory=None):
        self.bridge = bridge
        self.mosaic_factory = mosaic_factory
        self.lock = threading.Lock()
        self.mode = ROAD
        self.epoch = 0
        self.bridge_id = uuid.uuid4().hex
        self.pending = None
        self.peer = None
        self.adapter = None
        self.road_main = None
        self.road_specialist = None
        self.last_result_key = None

    def status(self, frame_id=None):
        with self.lock:
            value = {"mode": self.mode, "epoch": str(self.epoch),
                     "box_space": SPACES[self.mode], "bridge_id": self.bridge_id}
        if frame_id is not None:
            value["frame_id"] = int(frame_id)
        return value

    def token(self):
        with self.lock:
            return self.mode, self.epoch, self.bridge_id

    def reply(self, peer, state, epoch=None, message=""):
        value = self.status()
        value.update(state=state, message=message)
        if epoch is not None:
            value["request_epoch"] = str(epoch)
        payload = b"MIA1" + json.dumps(value, separators=(",", ":")).encode()
        self.bridge.result_socket.sendto(payload, peer)

    def receive(self, packet, peer):
        if packet[:4] != b"MVI1":
            return False
        if len(packet) != COMMAND.size:
            return True
        _, epoch, index = COMMAND.unpack(packet)
        if index not in (0, 1) or epoch == 0:
            self.reply(peer, "REJECTED", epoch, "invalid mode/epoch")
            return True
        mode = (ROAD, MOSAIC)[index]
        with self.lock:
            newest = self.pending[1] if self.pending else self.epoch
            if epoch < newest:
                state = "STALE"
            elif epoch == self.epoch:
                state = "APPLIED" if mode == self.mode else "REJECTED"
            elif self.pending and self.pending == (mode, epoch):
                state = "SWITCHING"
            else:
                self.pending, self.peer = (mode, epoch), peer
                state = "SWITCHING"
        self.reply(peer, state, epoch)
        return True

    def frame_request(self, packet):
        if packet[:4] not in (b"MVR1", b"MVR2", b"MVR3"):
            return packet  # Model/selection control must survive epoch synchronization.
        if packet[:4] != b"MVR3":
            # Legacy frame packets are accepted before the first handshake only.
            with self.lock:
                return packet if self.epoch == 0 and self.pending is None else b""
        if len(packet) != REQUEST.size:
            return b""
        _, frame_id, timestamp, epoch, input_fps, submitted_fps = REQUEST.unpack(packet)
        with self.lock:
            if epoch != self.epoch or self.pending is not None:
                return b""
        return struct.pack("<4sQQff", b"MVR2", frame_id, timestamp, input_fps, submitted_fps)

    def apply(self):
        with self.lock:
            if self.pending is None:
                return False
            mode, epoch = self.pending
            peer = self.peer
            self.pending = None
        try:
            if mode != self.mode:
                if mode == MOSAIC:
                    self.road_main = self.bridge.pipeline.main_detector
                    self.road_specialist = self.bridge.pipeline.specialist_detector
                    if self.mosaic_factory is None:
                        from fast_mosaic_backend import FastMosaicDetector
                        full_detector = FastMosaicDetector()
                    else:
                        full_detector = self.mosaic_factory(self.road_main)
                    self.adapter = MosaicRoadAdapter(full_detector, self.mosaic_factory is None)
                    self.bridge.pipeline.main_detector = self.adapter
                    self.bridge.pipeline.specialist_detector = None
                else:
                    self.bridge.pipeline.main_detector = self.road_main
                    self.bridge.pipeline.specialist_detector = self.road_specialist
                    self.adapter.close()
                    self.adapter = None
                    self.road_main = self.road_specialist = None
                self.bridge.pipeline.reset()
            with self.lock:
                self.mode, self.epoch = mode, epoch
            self.last_result_key = None
            self.bridge._model_options_dirty = True
            print("INFERENCE_MODE " + json.dumps(self.status(), separators=(",", ":")), flush=True)
            self.reply(peer, "APPLIED", epoch)
        except Exception as exc:
            self.reply(peer, "FAILED", epoch, str(exc))
            print("INFERENCE_MODE_FAILED", type(exc).__name__, str(exc), flush=True)
        return True

    def shutdown(self):
        if self.adapter is not None:
            self.bridge.pipeline.main_detector = self.road_main
            self.bridge.pipeline.specialist_detector = self.road_specialist
            self.adapter.close()
            self.adapter = None
