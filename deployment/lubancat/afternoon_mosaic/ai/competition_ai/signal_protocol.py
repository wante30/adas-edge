"""TLS1 draft offline encoder/decoder. Deliberately has no hardware transport."""
import struct
import zlib

HEADER = struct.Struct('<4sBBHIIQQHHI')  # 40 bytes
DIRECTION = struct.Struct('<BBBBHH')  # 8 bytes, three records
LAMP = struct.Struct('<IBBBB4HBBHHH')  # 24 bytes, up to eight records
COLORS = {'UNKNOWN': 0, 'RED': 1, 'YELLOW': 2, 'GREEN': 3}
SHAPES = {'UNKNOWN': 0, 'ROUND': 1, 'LEFT_ARROW': 2, 'STRAIGHT_ARROW': 3, 'RIGHT_ARROW': 4, 'UTURN_ARROW': 5}


def _score(value):
    return max(0, min(255, round(float(value)*255)))


def _age_remaining(row, now_ns):
    stamp = row.get('timestamp_ns', 0)
    expiry = row.get('expires_at_ns', 0)
    valid = bool(row.get('valid')) and 0 < stamp <= now_ns < expiry
    age = max(0, min(65535, (now_ns-stamp)//1_000_000))
    remaining = max(0, min(1200, (expiry-now_ns)//1_000_000)) if valid else 0
    return valid and remaining > 0, age, remaining


def encode_debug_packet(result, *, sequence, epoch, now_ns):
    if not 0 <= sequence <= 0xffffffff or not 0 <= epoch <= 0xffffffff:
        raise ValueError('sequence/epoch outside uint32')
    payload = bytearray()
    for code, key in enumerate(('left', 'straight', 'right'), 1):
        row = result.get('directions', {}).get(key, {})
        valid, age, remaining = _age_remaining(row, now_ns)
        color = COLORS.get(row.get('color'), 0) if valid else 0
        valid = valid and color != 0
        conflict = row.get('reason') == 'CONFLICTING_ASSOCIATED_COLORS'
        payload.extend(DIRECTION.pack(code, int(valid) | (int(conflict) << 1), color,
                                      _score(row.get('confidence', 0)) if valid else 0, age, remaining if valid else 0))
    signals = result.get('signals', [])[:8]
    for row in signals:
        valid, age, remaining = _age_remaining(row, now_ns)
        color = COLORS.get(row.get('color'), 0) if valid else 0
        shape = SHAPES.get(row.get('shape'), 0) if valid else 0
        # This flag reports evidence only, never grants a control permission.
        associated = row.get('association') == 'ASSOCIATED' and valid and now_ns < row.get('association_expires_at_ns', 0)
        box = [round(v) for v in row['bbox']]
        if not (0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 360):
            raise ValueError('TLS1 requires native 640x360 Q4 coordinates')
        payload.extend(LAMP.pack(row['track_id'], color, shape, int(valid) | (int(associated) << 1),
                       _score(row.get('confidence', 0)), *box,
                       _score(row.get('color_confidence', 0)), _score(row.get('shape_confidence', 0)),
                       age, remaining, 0))
    # flags bit0 MUST remain 1: draft/debug, no actuation.
    header = HEADER.pack(b'TLS1', 1, 1, HEADER.size, sequence, epoch, max(0, result.get('frame_id', 0)), now_ns,
                         len(payload), len(signals), 0)
    crc = zlib.crc32(header+payload)
    return header[:-4]+struct.pack('<I', crc)+payload


def decode_debug_packet(packet):
    if len(packet) < HEADER.size:
        raise ValueError('truncated header')
    magic, version, flags, size, sequence, epoch, frame_id, sent, length, count, crc = HEADER.unpack_from(packet)
    if (magic != b'TLS1' or version != 1 or flags != 1 or size != HEADER.size
            or count > 8 or length != 3*DIRECTION.size+count*LAMP.size
            or len(packet) != size+length):
        raise ValueError('unsupported header/length')
    if zlib.crc32(packet[:36]+b'\0\0\0\0'+packet[40:]) != crc:
        raise ValueError('CRC mismatch')
    directions = [DIRECTION.unpack_from(packet, size+i*DIRECTION.size) for i in range(3)]
    if any(code != i+1 or valid & ~3 or color > 3 or remaining > 1200
           or (valid & 1 and (color == 0 or remaining == 0))
           or (not valid & 1 and (color != 0 or remaining != 0)) or valid == 3
           for i, (code, valid, color, confidence, age, remaining) in enumerate(directions)):
        raise ValueError('invalid direction record')
    lamps = [LAMP.unpack_from(packet, size+24+i*LAMP.size) for i in range(count)]
    for tid, color, shape, valid, confidence, x1, y1, x2, y2, cc, sc, age, ttl, reserved in lamps:
        if (tid == 0 or color > 3 or shape > 5 or valid & ~3 or ttl > 1200 or reserved
                or (valid & 1 and ttl == 0) or (not valid & 1 and (color or shape or ttl or valid & 2))
                or not 0 <= x1 < x2 <= 640 or not 0 <= y1 < y2 <= 360):
            raise ValueError('invalid lamp record')
    if len({lamp[0] for lamp in lamps}) != len(lamps):
        raise ValueError('duplicate track IDs')
    return dict(sequence=sequence, epoch=epoch, frame_id=frame_id, sent_monotonic_ns=sent,
                directions=directions, lamps=lamps, debug_only=True)
