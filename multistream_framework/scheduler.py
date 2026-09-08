from __future__ import annotations

from typing import Mapping


class Scheduler:
    def select(self, stream_states: Mapping[int, Mapping[str, int]]) -> int:
        raise NotImplementedError


class RoundRobinScheduler(Scheduler):
    def __init__(self, stream_count: int) -> None:
        self.stream_count = stream_count
        self._cursor = 0

    def select(self, stream_states: Mapping[int, Mapping[str, int]]) -> int:
        for offset in range(self.stream_count):
            stream_id = (self._cursor + offset) % self.stream_count
            if stream_states.get(stream_id, {}).get("queue_length", 0) > 0:
                self._cursor = (stream_id + 1) % self.stream_count
                return stream_id
        return -1
