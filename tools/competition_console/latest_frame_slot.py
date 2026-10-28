"""Condition-backed single slot for work that should use only the newest frame."""
from threading import Condition


class LatestFrameSlot:
    def __init__(self):
        self._condition = Condition()
        self._value = None
        self._closed = False
        self.replaced_count = 0

    def publish(self, value):
        with self._condition:
            if self._closed:
                return False
            replaced = self._value is not None
            if replaced:
                self.replaced_count += 1
            self._value = value
            self._condition.notify()
            return replaced

    def take(self, timeout=None):
        with self._condition:
            if self._value is None and not self._closed:
                self._condition.wait_for(
                    lambda: self._value is not None or self._closed, timeout=timeout
                )
            value, self._value = self._value, None
            return value

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
