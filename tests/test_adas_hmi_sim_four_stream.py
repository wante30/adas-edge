from tools.adas_hmi_sim.backend.video_pipeline import FourStreamMockSource


def test_pc_system_simulates_four_synchronised_channels_and_wire_ids():
    source = FourStreamMockSource(start_ms=4000)
    packet = source.next()

    assert packet is not None
    assert sorted(packet.stream_images) == ["front", "left", "rear", "right"]
    assert packet.metadata["system_mode"] == "PC_FOUR_STREAM"
    assert packet.metadata["pcie"]["wire_protocol"] == "ADSF/ADSR v1"
    assert packet.metadata["pcie"]["descriptor_order"] == [0, 1, 2, 3]
    assert packet.metadata["rk3566"] == {"device": "RK3566_SIM", "detector": "MOCK", "stream_count": 4}
    assert packet.transport["pcie_bytes"] > 0
    assert packet.transport["result_bytes"] > 0

    source.close()
