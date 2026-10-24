import numpy as np

from tools.b05k_video_benchmark import iter_video_frames


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None


def test_warmup_frame_is_processed_once_without_rewinding_capture():
    frames = [np.full((1, 1, 3), value, dtype=np.uint8) for value in range(3)]
    capture = FakeCapture(frames[1:])

    observed = [int(frame[0, 0, 0]) for frame in iter_video_frames(capture, frames[0])]

    assert observed == [0, 1, 2]
