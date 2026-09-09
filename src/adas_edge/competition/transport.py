from __future__ import annotations

from collections import deque


class RoundRobinTransport:
    def __init__(self, streams: int = 4, queue_depth: int = 3) -> None:
        self.queues = [deque(maxlen=queue_depth) for _ in range(streams)]
        self.next_stream = 0
        self.dropped = [0] * streams

    def push(self, frame) -> None:
        if not 0 <= frame.stream_id < len(self.queues):
            raise ValueError(f"stream_id out of range: {frame.stream_id}")
        if len(self.queues[frame.stream_id]) == self.queues[frame.stream_id].maxlen:
            self.queues[frame.stream_id].popleft()
            self.dropped[frame.stream_id] += 1
        self.queues[frame.stream_id].append(frame)

    def pop(self):
        for offset in range(len(self.queues)):
            sid = (self.next_stream + offset) % len(self.queues)
            if self.queues[sid]:
                self.next_stream = (sid + 1) % len(self.queues)
                return self.queues[sid].popleft()
        return None
