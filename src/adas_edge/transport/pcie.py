from adas_edge.transport.control import ControlTransport


class PCIeControlTransport(ControlTransport):
    """Contract placeholder. Requires a negotiated FPGA command-ring specification."""

    def send(self, command):  # type: ignore[no-untyped-def]
        raise NotImplementedError("PCIe control transport is intentionally out of Phase 1")
