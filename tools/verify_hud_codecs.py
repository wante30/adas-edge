"""Build portable C codecs and verify byte-for-byte Python interoperability."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from competition_io.adas_status_v1 import Action, AdasStatus, ClockDomain, Lane, Light, Valid, Warning
from competition_io.hud_v1 import VECTORS


def verify(output: Path, compiler: str | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    compiler = compiler or shutil.which("gcc")
    if not compiler:
        raise RuntimeError("Pass the installed C compiler with --cc")
    env = os.environ.copy()
    # MSYS2's native compiler/helper DLLs must be on the child PATH.
    compiler_dir = str(Path(compiler).resolve().parent)
    env["PATH"] = compiler_dir + os.pathsep + env.get("PATH", "")
    executable = output.resolve() / ("codec_selftest.exe" if os.name == "nt" else "codec_selftest")
    command = [compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic",
               str(ROOT / "deployment/hud/codec_selftest.c"), "-o", str(executable)]
    compiled = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    c_message = AdasStatus(0x12345678, 3, 0x1122334455667788, 0x0102030405060708,
                          Light.GREEN, Lane.DASHED, Lane.SOLID, Action.LEFT,
                          26214/32767, ClockDomain.VIDEO_REPLAY, Valid(31), Warning.OVERSPEED)
    python_message = AdasStatus(93, 2, 0xFEDCBA9876543210, 10123456789, Light.RED,
                               action=Action.STOP, signal_confidence=.9,
                               clock_domain=ClockDomain.FPGA_SOURCE_NS,
                               valid=Valid.SIGNAL_OBSERVED | Valid.SIGNAL_ASSOCIATED | Valid.DECISION,
                               warnings=Warning.RED_LIGHT, valid_for_ms=333)
    packet_path = output.resolve() / "python_interop_packet.bin"
    packet_path.write_bytes(python_message.serialize())
    result = subprocess.run([str(executable), str(packet_path)], env=env, capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    words = [int(line.split()[1], 16) for line in lines if line.startswith("HUD ")]
    assert words == [word for _, word in VECTORS]
    c_packet = bytes.fromhex(next(line[5:] for line in lines if line.startswith("ADST ")))
    assert c_packet == c_message.serialize() == AdasStatus.deserialize(c_packet).serialize()
    assert f"PYTHON_DECODE 93 2 {python_message.source_frame_id}" in lines
    assert "NATIVE_C_SELFTEST_PASS" in lines
    (output / "native_codec_output.txt").write_text(result.stdout, encoding="utf-8")
    report = {"evidence": "PC_NATIVE_C_PYTHON_CODEC_INTEROPERABILITY_NOT_ARM_OR_FPGA",
              "passed": True, "compiler": compiler, "compile_command": command,
              "compiler_diagnostics": compiled.stdout + compiled.stderr,
              "hud_vectors": len(words), "c_to_python": True, "python_to_c": True,
              "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}
    (output / "codec_interop.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.cc), indent=2))
