from __future__ import annotations
from collections import defaultdict, deque
from .protocol import FrameDescriptor, ResultDescriptor

class MockTransport:
    """In-memory FPGA↔RK3566 transport; stream and frame IDs are retained."""
    def __init__(self): self.frames=defaultdict(deque); self.results=defaultdict(deque)
    def send_frame(self, packet:FrameDescriptor): self.frames[packet.stream_id].append(packet.serialize())
    def recv_frame(self, stream_id:int): return FrameDescriptor.deserialize(self.frames[stream_id].popleft()) if self.frames[stream_id] else None
    def send_result(self, packet:ResultDescriptor): self.results[packet.stream_id].append(packet.serialize())
    def recv_result(self, stream_id:int): return ResultDescriptor.deserialize(self.results[stream_id].popleft()) if self.results[stream_id] else None
