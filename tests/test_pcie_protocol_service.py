import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "deployment" / "lubancat" / "qt_gpu_probe"
sys.path.insert(0, str(ROOT))

from pcie_protocol_service import DebugBackend, SPEC  # noqa: E402


def backend():
    return DebugBackend("/nonexistent", simulation=True)


def test_register_map_unique_aligned_whitelisted():
    regs = SPEC["registers"]
    assert len(regs) == 123
    offsets = [r["offset_bytes"] for r in regs]
    assert len(set(offsets)) == 123
    assert all(o % 4 == 0 for o in offsets)
    assert all(r["bar"] == 1 for r in regs)


def test_hud_golden_vectors_and_blink_guard():
    b = backend()
    r = b.raw_hud(0x00011103)
    assert r["accepted"] == 1 and r["applied"] == 1
    assert r["effective"] == "0x00000103"
    r = b.raw_hud(0x00020000)
    assert r["accepted"] == 2
    r = b.hud_apply({"sequence": 3, "fields": {"left_blink": 1}})
    assert not r["ok"] and r["state"] == "REJECTED"


def test_hud_sequence_and_historical_error():
    b = backend()
    r = b.hud_apply({"sequence": 0, "fields": {"hud_enable": 1, "left": 1}})
    assert r["ok"] and r["sequence"] == 1
    # A rejected write sets LAST_ERROR; a later successful write must preserve it.
    r = b.hud_apply({"sequence": 2, "fields": {"left_blink": 1}})
    assert not r["ok"] and b.hud_read()["last_error"] == 4
    r = b.hud_apply({"sequence": 3, "fields": {"hud_enable": 1}})
    assert r["ok"]
    assert b.hud_read()["last_error"] == 4


def test_scene_vector_bcd_and_clear():
    b = backend()
    r = b.raw_scene({"writes": {"0x638": "0x985", "0x63c": "0x10242026",
                                "0x640": "0x53014", "0x644": "0x2a"}})
    assert r["applied"] == 42
    assert b.scene_read()["time0"] == "0x10242026"
    r = b.scene_apply({"fields": {"valid": 0}, "sequence": 43})
    assert r["ok"] and r["state"] == "ACCEPTED"
    assert b.scene_read()["shadow"] == "0x00000000"


def test_nav_vector_and_clear():
    b = backend()
    r = b.nav_apply({"enable": 1, "direction": 1, "distance": 120, "sequence": 1})
    assert r["shadow"] == "0x00780003"
    assert r["control"] == "0x00010001"
    assert r["active"] == "0x00780003"
    c = b.nav_clear()
    assert c["ok"]


def test_isp_fields_roundtrip_and_signed_offset():
    b = backend()
    r = b.isp_apply({"global": 30, "tiles": [{"gain_enable": 1, "gain_r": 300,
                                              "offset_r": -7}, {}, {}, {}]})
    assert r["ok"]
    q = b.isp_query()
    f = q["tile_fields"][0]
    assert f["gain_enable"] == 1 and f["gain_r"] == 300 and f["offset_r"] == -7
    # The high enable/field bits round-trip without changing dword ordering.
    assert len(q["tiles"][0]) == 4


def test_algo_bits_56_to_59_and_query():
    b = backend()
    r = b.algo_apply({"fields": {"stage_mask": 0b0101, "tile_mask": 15,
                                 "w": 230, "t_min": 25, "atmosphere_a": 220,
                                 "flags": 1, "threshold": 128}})
    assert r["ok"]
    q = b.algo_query()
    assert q["fields"]["stage_mask"] == 0b0101
    assert (int(q["value"], 16) >> 56) & 0xFF == 0b0101


def test_diag_and_dma_are_read_only():
    b = backend()
    d = b.diag()
    assert d["ok"]
    assert d["dma_write"] == "BLOCKED"
    assert d["hsst"]["protocol"] == "0x4c4d0001"
    assert d["hsst"]["left_count"] == 2


def test_concurrent_hud_sequence_serialized():
    b = backend()
    errors = []

    def run(seq):
        try:
            b.hud_apply({"sequence": seq, "fields": {"hud_enable": 1}})
        except Exception as exc:
            errors.append(exc)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(2, 12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    accepted = b.hud_read()["accepted"]
    assert accepted in range(2, 12)
