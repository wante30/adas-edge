"""Package the local HUD/ADST SDK and explicit PC evidence using an allowlist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "competition_io/__init__.py", "competition_io/protocol.py",
    "competition_io/mock_transport.py", "competition_io/multistream.py",
    "competition_io/hud_v1.py", "competition_io/hud_client.py",
    "competition_io/hud_simulator.py", "competition_io/adas_status_v1.py",
    "competition_ai/__init__.py", "competition_ai/scene.py", "competition_ai/adas.py",
    "deployment/hud/hud_v1.h", "deployment/hud/adas_status_v1.h", "deployment/hud/codec_selftest.c",
    "tools/hud_control_demo.py", "tools/export_adas_status.py", "tools/verify_hud_codecs.py",
    "tools/package_hud_handoff.py", "tests/test_hud_v1_encoding.py",
    "tests/test_hud_transactions.py", "tests/test_adas_status_v1.py", "tests/test_hud_pc_tools.py",
    "docs/pcie_hud_pc_handoff.md",
)
EVIDENCE_FILES = (
    "validation_summary.json", "pytest.txt", "codec_interop.json", "native_codec_output.txt",
    "hud_demo.json", "hud_frame_trace.csv", "adas_export.json", "official_video_adst.bin",
    "official_video_adst.jsonl", "python_interop_packet.bin",
)
README = """# HUD / ADST PC handoff

This package contains locally tested SDK code and PC evidence. It contains no
FPGA bitstream, driver ioctl binding, RKNN model or hardware validation.
The existing competition policy and descriptor modules are included only as
the SDK's source dependencies; they were not changed in this task.

Use Python 3.8+ (tested locally with 3.11); SDK and CLI tools need only the
standard library. Tests need pytest. Run from this extracted directory:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q tests --basetemp outputs/pytest_run
python tools/hud_control_demo.py --output outputs/demo
python tools/verify_hud_codecs.py --cc C:/msys64/ucrt64/bin/gcc.exe --output outputs/codecs
```

On Linux use `--cc gcc` for the last command. C codecs are userspace C99.
Compiler/platform support on a different machine must be verified there.
`export_adas_status.py` requires the original predictions.jsonl; it is not
included. `evidence/official_video_adst.bin` contains the already exported
60-byte records with CRC. It uses VIDEO_REPLAY time, and must not be sent as
FPGA clock-domain data by changing a tag.

Read `docs/pcie_hud_pc_handoff.md` for the wire contract, semantics, driver
adapter requirements and remaining hardware checks. `evidence/` reports
preserve the paths and commands used on the original PC. MANIFEST.json
contains SHA256 for every included file except itself.
"""


def package(evidence: Path, archive: Path) -> dict:
    import pytest
    files = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    files.update({"evidence/"+name: (evidence / name).read_bytes() for name in EVIDENCE_FILES})
    files["README.md"] = README.encode("utf-8")
    files["requirements-test.txt"] = f"pytest=={pytest.__version__}\n".encode("utf-8")
    manifest = {"evidence": "LOCAL_SOFTWARE_HANDOFF_NOT_HARDWARE_IMPLEMENTATION",
                "python_tested": sys.version, "files": [
                    {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in sorted(files.items())]}
    files["MANIFEST.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for name, data in sorted(files.items()):
            handle.writestr(name, data)
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        for entry in manifest["files"]:
            assert hashlib.sha256(handle.read(entry["path"])).hexdigest() == entry["sha256"]
    result = {"archive": str(archive.resolve()), "file_count": len(files),
              "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    archive.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.evidence, args.output), indent=2))
