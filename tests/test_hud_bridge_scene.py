import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "deployment" / "lubancat" / "qt_gpu_probe"
sys.path.insert(0, str(ROOT))

from hud_bridge import (  # noqa: E402
    Bridge,
    OFF_EXT_APPLIED,
    OFF_EXT_COMMIT,
    OFF_EXT_REQUEST,
    OFF_EXT_STATUS,
    OFF_ERROR,
    OFF_REJECT,
    OFF_TIME0,
    OFF_TIME1,
)


class FakeBridge(Bridge):
    def __init__(self):
        self.ext_seq = 0
        self.legacy_seq = 0
        self.values = {
            OFF_EXT_STATUS: 2,
            OFF_EXT_APPLIED: 0,
            OFF_TIME0: 0x18102026,
            OFF_TIME1: 0x00183014,
            OFF_ERROR: 0,
            OFF_REJECT: 0,
        }
        self.writes = []

    def rd(self, offset):
        return self.values.get(offset, 0)

    def wr(self, offset, value):
        self.writes.append((offset, value))
        self.values[offset] = value & 0xFFFFFFFF
        if offset == OFF_EXT_COMMIT:
            self.values[OFF_EXT_APPLIED] = value & 0xFFFF
            self.values[OFF_EXT_STATUS] = 2


def test_scene_commit_without_time_update_keeps_previous_time_words():
    bridge = FakeBridge()

    result = bridge.set({
        "perception_enable": True,
        "traffic": 1,
        "time_enable": False,
    })

    assert result["ok"]
    assert result["time0"] == 0x18102026
    assert result["time1"] == 0x00183014
    assert [offset for offset, _value in bridge.writes] == [OFF_EXT_REQUEST, OFF_EXT_COMMIT]
