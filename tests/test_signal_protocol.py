import pytest
from competition_ai.signal_protocol import encode_debug_packet, decode_debug_packet, HEADER, LAMP
from test_signal_shape_association import signal, binding, result
from competition_ai.signal_association import associate


def test_draft_packet_sizes_crc_and_remaining_ttl():
    data = associate(result([signal()]), now_ns=1_100_000_000, bindings=[binding()])
    encoded = encode_debug_packet(data, sequence=42, epoch=5, now_ns=1_100_000_000)
    assert HEADER.size == 40 and LAMP.size == 24 and len(encoded) == 88
    decoded = decode_debug_packet(encoded)
    assert decoded['debug_only'] and decoded['sequence'] == 42
    assert decoded['directions'][0][2] == 3  # green
    assert decoded['directions'][0][-1] == 400  # road binding expires before lamp
    with pytest.raises(ValueError, match='CRC'):
        decode_debug_packet(encoded[:-1]+bytes([encoded[-1] ^ 1]))


def test_expired_green_cannot_renew_and_lengths_are_checked():
    data = associate(result([signal()]), now_ns=1_100_000_000, bindings=[binding()])
    packet = encode_debug_packet(data, sequence=0, epoch=1, now_ns=2_500_000_000)
    decoded = decode_debug_packet(packet)
    assert decoded['directions'][0][2] == 0
    assert decoded['lamps'][0][1] == 0
    for truncated in (packet[:5], packet[:-1], packet+b'extra'):
        with pytest.raises(ValueError):
            decode_debug_packet(truncated)
