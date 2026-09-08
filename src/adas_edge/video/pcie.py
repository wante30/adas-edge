from adas_edge.video.base import FrameSource


class PCIeFrameSource(FrameSource):
    """Contract placeholder. Requires FPGA BAR/DMA and buffer-lifetime specification."""

    def read(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError("PCIe DMA is intentionally out of Phase 1")

    def close(self) -> None:
        raise NotImplementedError("PCIe DMA is intentionally out of Phase 1")
