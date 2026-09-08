from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from .protocol import FrameDescriptor, ResultDescriptor

@dataclass
class MockRoundRobin:
    streams: int = 4
    queue_depth: int = 3

    def __post_init__(self):
        self.queues = [deque(maxlen=self.queue_depth) for _ in range(self.streams)]
        self.next_stream = 0
        self.dropped = [0] * self.streams

    def push(self, frame: FrameDescriptor) -> None:
        if not 0 <= frame.stream_id < self.streams:
            raise ValueError(f"stream_id out of range: {frame.stream_id}")
        if len(self.queues[frame.stream_id]) == self.queue_depth:
            self.queues[frame.stream_id].popleft(); self.dropped[frame.stream_id] += 1
        self.queues[frame.stream_id].append(frame)

    def pop(self) -> FrameDescriptor | None:
        for offset in range(self.streams):
            sid = (self.next_stream + offset) % self.streams
            if self.queues[sid]:
                self.next_stream = (sid + 1) % self.streams
                return self.queues[sid].popleft()
        return None
