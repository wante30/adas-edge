from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\ai_bridge.py')
s=p.read_text(encoding='utf-8')
# insert PackedRgb565 after decode_rgb565_payload
needle='''def decode_rgb565_payload(payload: bytes):
    if len(payload) != FRAME_BYTES:
        raise ValueError(f"RGB565 frame must be {FRAME_BYTES} bytes, got {len(payload)}")
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)
    return cv2.cvtColor(packed, cv2.COLOR_BGR5652BGR)
'''
rep=needle+'''

class PackedRgb565:
    """A frame kept in packed form; decode only the quadrants a model needs."""
    __slots__ = ("payload",)

    def __init__(self, payload: bytes):
        if len(payload) != FRAME_BYTES:
            raise ValueError(f"RGB565 frame must be {FRAME_BYTES} bytes, got {len(payload)}")
        self.payload = payload

    def quadrant_bgr(self, quadrant: int):
        if quadrant not in (0, 1, 2, 3):
            raise ValueError("quadrant must be 0, 1, 2, or 3")
        row, col = divmod(quadrant, 2)
        packed = np.frombuffer(self.payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)
        roi = packed[row * HEIGHT // 2:(row + 1) * HEIGHT // 2,
                     col * WIDTH // 2:(col + 1) * WIDTH // 2]
        return cv2.cvtColor(roi, cv2.COLOR_BGR5652BGR)

    def full_bgr(self):
        packed = np.frombuffer(self.payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)
        return cv2.cvtColor(packed, cv2.COLOR_BGR5652BGR)
'''
if needle not in s: raise SystemExit('decode marker')
s=s.replace(needle,rep,1)
# listener: no full BGR decode
s=s.replace('''                try:
                    frame = decode_rgb565_payload(payload)
                except ValueError:
                    self.shared_frame_mismatches += 1
                    continue''','''                try:
                    frame = PackedRgb565(payload)
                except ValueError:
                    self.shared_frame_mismatches += 1
                    continue''',1)
# _analyze quadrant extraction
s=s.replace('''        if full_frame is None:
            full_frame = self.frames[frame_id % len(self.frames)]
        if self.args.layout == "quad":
            road = crop_quadrant(full_frame, self.args.road_quadrant)
            navigation_image = crop_quadrant(full_frame, self.args.navigation_quadrant)
        else:
            road = full_frame
            navigation_image = full_frame''','''        if full_frame is None:
            full_frame = self.frames[frame_id % len(self.frames)]
        if isinstance(full_frame, PackedRgb565):
            if self.args.layout == "quad":
                road = full_frame.quadrant_bgr(self.args.road_quadrant)
                navigation_image = full_frame.quadrant_bgr(self.args.navigation_quadrant)
            else:
                road = full_frame.full_bgr()
                navigation_image = road
        elif self.args.layout == "quad":
            road = crop_quadrant(full_frame, self.args.road_quadrant)
            navigation_image = crop_quadrant(full_frame, self.args.navigation_quadrant)
        else:
            road = full_frame
            navigation_image = full_frame''',1)
p.write_text(s,encoding='utf-8')
print('PackedRgb565 ROI decode installed')
