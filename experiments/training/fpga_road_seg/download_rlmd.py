"""Recover the public RLMD archive, preserving provenance and TLS validation."""
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(r'C:\Users\JingyuWang\datasets\rlmd_issue15')
REVISION = 'a1f0fdbcb3f56c0a6cc999c7dbf860d67246bbb0'
NAME = 'RLMD_1080p-20250613T030116Z-1-001.zip'
SHA256 = '1269c85675a04fbb0f48cb48517d2fd8e8f8614b7364146f28355b19dee845c4'
SIZE = 1356808912
URL = f'https://huggingface.co/datasets/veetinator/Road_Line_Marking_Dataset/resolve/{REVISION}/{NAME}'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.proxies = {'https': 'http://127.0.0.1:7890'}
    sources = {
        'official_readme.md': 'https://raw.githubusercontent.com/stu9113611/RLMD/main/README.md',
        'official_classes.csv': 'https://raw.githubusercontent.com/stu9113611/RLMD/main/classes.csv',
        'official_repository.json': 'https://api.github.com/repos/stu9113611/RLMD',
        'official_folder.html': 'https://drive.google.com/drive/folders/18-Lnz2MaPQWSOJ8aK1ikkwXLLdjImYOA',
        'mirror_metadata.json': 'https://huggingface.co/api/datasets/veetinator/Road_Line_Marking_Dataset',
    }
    for name, url in sources.items():
        target = ROOT/name
        if not target.exists():
            response = session.get(url, timeout=45)
            response.raise_for_status()
            target.write_bytes(response.content)
    dest = ROOT/NAME
    if not dest.exists():
        if shutil.disk_usage(ROOT).free < SIZE*3:
            raise OSError('insufficient disk space for archive and extraction')
        partial = dest.with_suffix('.partial')
        for attempt in range(1, 13):
            start = partial.stat().st_size if partial.exists() else 0
            if start == SIZE:
                break
            try:
                with session.get(URL, headers={'Range': f'bytes={start}-'} if start else {},
                                 stream=True, timeout=(30,90)) as response:
                    response.raise_for_status()
                    if start and (response.status_code != 206 or
                                  not response.headers.get('Content-Range','').startswith(f'bytes {start}-')):
                        raise IOError('server did not honor resume range')
                    with open(partial, 'ab' if start else 'wb') as stream:
                        for block in response.iter_content(4*1024*1024):
                            stream.write(block)
                            if stream.tell()//(64*1024*1024) != (stream.tell()-len(block))//(64*1024*1024):
                                print(f'downloaded={stream.tell()}/{SIZE}', flush=True)
                break
            except requests.RequestException as exc:
                print(f'attempt={attempt} {type(exc).__name__}: {exc}', flush=True)
                if attempt == 12:
                    raise
                time.sleep(5)
        if partial.stat().st_size != SIZE or sha256(partial) != SHA256:
            raise IOError('RLMD archive failed size/SHA256 check')
        os.replace(partial, dest)
    if dest.stat().st_size != SIZE or sha256(dest) != SHA256:
        raise IOError('existing RLMD archive failed size/SHA256 check')
    extraction = ROOT/'extracted'
    with zipfile.ZipFile(dest) as archive:
        for entry in archive.infolist():
            target = (extraction/entry.filename).resolve()
            if not target.is_relative_to(extraction.resolve()):
                raise ValueError('unsafe archive path')
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                if target.stat().st_size != entry.file_size:
                    raise IOError(f'existing extracted member differs: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, open(target,'wb') as output:
                shutil.copyfileobj(source, output)
    record = dict(official='https://github.com/stu9113611/RLMD', official_download=sources['official_folder.html'],
                  mirror=URL, revision=REVISION, size=SIZE, sha256=SHA256,
                  license='Official public research dataset with citation request; no explicit license in official repo. Mirror declares MIT; this is an uploader claim, not a verified grant by original authors. Local academic experiment only; no redistribution of data.',
                  verification='SHA256 against pinned mirror LFS; official class CSV and folder checked separately',
                  extraction=str(extraction))
    (ROOT/'provenance.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2), flush=True)


if __name__ == '__main__':
    main()
