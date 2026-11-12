"""Download the public original BDD100K zip archives with resume and MD5 checks."""
import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


FILES = {
    'bdd100k_images.zip': (6_941_520_264, 'b538e3731a132e28dae37f18c442c51e'),
    'bdd100k_labels.zip': (154_084_208, '035bd9e56c0c2dbdef2165e0bf1cf6a5'),
}
BASE = 'https://archive.org/download/bdd100k/'
CHUNK_SIZE = 32 * 1024 * 1024


def digest(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(name, out, workers=8, attempts=12):
    expected_size, expected_md5 = FILES[name]
    dest = Path(out) / name
    if dest.is_file() and dest.stat().st_size == expected_size and digest(dest) == expected_md5:
        print(f'{name}: already verified size={expected_size} md5={expected_md5}', flush=True)
        return dict(path=str(dest.resolve()), size=expected_size, md5=expected_md5, source=BASE + name)
    if shutil.disk_usage(out).free < 2 * expected_size + 1024**3:
        raise OSError('insufficient disk space for resumable chunks and verified archive assembly')
    chunk_dir = Path(out) / f'.{name}.chunks'
    chunk_dir.mkdir(exist_ok=True)
    chunks = [(start, min(start + CHUNK_SIZE, expected_size) - 1)
              for start in range(0, expected_size, CHUNK_SIZE)]

    def fetch(part):
        start, end = part
        chunk = chunk_dir / f'{start:012d}.part'
        size = end - start + 1
        if chunk.exists() and chunk.stat().st_size == size:
            return
        for attempt in range(1, attempts + 1):
            try:
                with requests.get(BASE + name, headers={'Range': f'bytes={start}-{end}'},
                                  stream=True, timeout=(30, 180)) as response:
                    expected_range = f'bytes {start}-{end}/{expected_size}'
                    if response.status_code != 206 or response.headers.get('Content-Range', '') != expected_range:
                        raise IOError(f'expected byte range {start}-{end}, got HTTP {response.status_code} {response.headers.get("Content-Range")}')
                    tmp = chunk.with_suffix('.tmp')
                    received = 0
                    with open(tmp, 'wb') as f:
                        for block in response.iter_content(1024 * 1024):
                            if block:
                                f.write(block)
                                received += len(block)
                    if received != size:
                        raise IOError(f'chunk size mismatch {received} != {size}')
                    tmp.replace(chunk)
                    return
            except Exception as exc:
                print(f'{name} range={start}-{end} attempt={attempt}/{attempts}: {type(exc).__name__}: {exc}', flush=True)
                if attempt == attempts:
                    raise
                time.sleep(min(attempt * 2, 20))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, part) for part in chunks]
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 8 == 0 or done == len(chunks):
                print(f'{name}: chunks={done}/{len(chunks)}', flush=True)

    partial = dest.with_suffix(dest.suffix + '.partial')
    with open(partial, 'wb') as output:
        for start, _ in chunks:
            with open(chunk_dir / f'{start:012d}.part', 'rb') as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
    if partial.stat().st_size != expected_size:
        raise IOError(f'{name} size mismatch: {partial.stat().st_size} != {expected_size}')
    actual_md5 = digest(partial)
    if actual_md5 != expected_md5:
        raise IOError(f'{name} MD5 mismatch: {actual_md5} != {expected_md5}')
    partial.replace(dest)
    print(f'{name}: verified size={expected_size} md5={actual_md5}', flush=True)
    return dict(path=str(dest.resolve()), size=expected_size, md5=actual_md5, source=BASE + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--only', choices=['images', 'labels', 'both'], default='both')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    names = ['bdd100k_labels.zip', 'bdd100k_images.zip'] if args.only == 'both' else [f'bdd100k_{args.only}.zip']
    result = {}
    for name in names:
        result[name] = download(name, root, workers=args.workers)
        (root / 'download_manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
