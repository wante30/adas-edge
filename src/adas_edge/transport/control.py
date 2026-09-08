from __future__ import annotations

from abc import ABC, abstractmethod
import json
import socket
from typing import TextIO

from adas_edge.decision.engine import ControlCommand


def command_payload(command: ControlCommand) -> dict[str, object]:
    return {
        "action": command.action,
        "warning_flags": list(command.warning_flags),
        "confidence": command.confidence,
        "timestamp": command.timestamp,
    }


class ControlTransport(ABC):
    @abstractmethod
    def send(self, command: ControlCommand) -> None:
        """Transmit an implementation-neutral control command."""


class ConsoleControlTransport(ControlTransport):
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def send(self, command: ControlCommand) -> None:
        self._stream.write(json.dumps(command_payload(command), ensure_ascii=False) + "\n")
        self._stream.flush()


class UDPControlTransport(ControlTransport):
    def __init__(self, host: str, port: int) -> None:
        self._address = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, command: ControlCommand) -> None:
        self._socket.sendto(json.dumps(command_payload(command)).encode("utf-8"), self._address)

    def close(self) -> None:
        self._socket.close()
