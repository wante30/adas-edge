"""Create a deterministic, separate-directory ARM source/debug bundle.

No deployment command, startup edit, model download, or hardware write occurs.
Installed RKNN models/runtime and libp2capi_generic.so are external prerequisites.
"""
import argparse
import hashlib
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for source, target in [('deployment/lubancat/afternoon_mosaic/ai', 'ai'),
                           ('deployment/lubancat/afternoon_mosaic/hmi', 'hmi'),
                           ('deployment/lubancat/afternoon_mosaic/fast_mosaic', 'fast_mosaic'),
                           ('competition_ai', 'competition_ai'), ('docs/multi_signal', 'docs'),
                           ('tests/fixtures/real_signals', 'tests/fixtures/real_signals')]:
        for path in (ROOT/source).rglob('*'):
            if (path.is_file() and '__pycache__' not in path.parts
                    and path.suffix not in ('.so', '.rknn', '.onnx', '.pyc')):
                files[str(Path(target)/path.relative_to(ROOT/source)).replace('\\', '/')] = path
    for name in ('check_signal_backend.py', 'evaluate_signal_photos.py', 'collect_signal_photos.py',
                 'replay_multi_signals.py', 'board_multi_signal_ab.py'):
        files['tools/'+name] = ROOT/'tools'/name
    manifest = dict(schema_version=1, debug_only=True,
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        baseline_tag='rollback/multi-signals-board-20261111',
        files={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())})
    archive = args.output/'adas-multi-signals-arm-debug.tgz'
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed, tarfile.open(fileobj=compressed, mode='w') as tar:
        for name, path in sorted(files.items()):
            info = tar.gettarinfo(str(path), 'adas-multi-signals-debug/'+name)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mode = 0o755 if path.suffix == '.sh' else 0o644
            with path.open('rb') as stream:
                tar.addfile(info, stream)
        data = json.dumps(manifest, indent=2).encode('utf-8')
        info = tarfile.TarInfo('adas-multi-signals-debug/bundle_manifest.json')
        info.size = len(data); info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    (args.output/'bundle_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.output/'SHA256SUMS').write_text(digest+'  '+archive.name+'\n', encoding='utf-8')
    print(json.dumps(dict(archive=str(archive.resolve()), bytes=archive.stat().st_size,
                          files=len(files), dirty=manifest['dirty'], sha256=digest)))


if __name__ == '__main__':
    main()
