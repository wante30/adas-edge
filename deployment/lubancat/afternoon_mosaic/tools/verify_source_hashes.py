"""Verify source provenance after copying between Windows and the RK3566 board."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'config/verified_source_hashes.json'
def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n').rstrip()+b'\n').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    expected=json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
    if a.write:
        for folder in ('ai','hmi','tools','tests','hardware','fast_mosaic'):
            for path in (ROOT/folder).rglob('*'):
                if path.is_file() and path.suffix in ('.py','.cpp','.h','.qml','.pro','.qrc','.sh') and '__pycache__' not in path.parts and not path.name.startswith('diagnostic_') and not path.name.endswith('_static_tesla.qml'):
                    expected[path.relative_to(ROOT).as_posix()]=digest(path)
        expected={key:digest(ROOT/key) for key in sorted(expected)}
        MANIFEST.write_text(json.dumps(expected,indent=2)+'\n',encoding='utf-8')
    errors=[key for key,value in expected.items() if not (ROOT/key).is_file() or digest(ROOT/key)!=value]
    print(json.dumps({'verified':len(expected)-len(errors),'mismatches':errors}))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
