"""Reproducible board runtime archive; omit sessions, recordings and external models."""
import argparse,hashlib,json,platform,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"runtime/adas-risk-control-rk3566.zip")
    args=parser.parse_args();out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name in ("ai","hmi","fast_mosaic","tools","tests","validation","config"):
        paths.extend(p for p in (ROOT/name).rglob("*") if p.is_file() and not any(
            part in ("__pycache__",".git") or part.startswith("build-") for part in p.relative_to(ROOT).parts)
            and p.suffix not in (".pyc",".o",".xwd") and not p.name.endswith((".bak",".pre-audit")))
    paths.extend(ROOT.glob("*.md"));paths.append(ROOT/"baseline_hashes.json")
    binary=ROOT/"build-stage4/qt_gpu_probe";library=ROOT/"fast_mosaic/capi/libp2capi_generic.so"
    for p in (binary,library):
        if not p.is_file():raise FileNotFoundError(str(p))
        if p not in paths:paths.append(p)
    files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))}
    manifest=dict(schema="ADAS_FOLLOWUP_RUNTIME_V1",source_commit=args.source_commit,
        base_commit="d397af411d592447197fa2f194144dca52c765dd",architecture=platform.machine(),
        default_control_mode="DRY_RUN",external_dependencies=[
            "RK3566 aarch64 Linux / Qt5 / Mali EGL / installed RKNN runtime",
            "Python3.8+, numpy, OpenCV, Pillow and existing OCR environment",
            "Original B, B-capi, MTL-FULL, MTL-FAST and BDD model assets (see ai/model_profiles.py)",
            "Optional 1350-frame 1280x720 RGB565 recording (not included)",
            "Bundled config/baseline_launch.json supports this board without a connected PCIe endpoint"],files=files)
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in files:archive.write(ROOT/name,name)
        archive.writestr("RUNTIME_MANIFEST.json",json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out) as archive:assert archive.testzip() is None
    checksum=sha(out);out.with_suffix(out.suffix+".sha256").write_text(checksum+"  "+out.name+"\n",encoding="utf-8")
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,sha256=checksum,files=len(files))),flush=True)
if __name__=="__main__":main()
