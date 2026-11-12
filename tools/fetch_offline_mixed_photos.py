"""Restore exact new real-photo evaluation inputs; preserve any existing file."""
import argparse
import hashlib
import json
from pathlib import Path
import requests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('docs/multi_signal/offline_mixed_photos.json'))
    args=parser.parse_args()
    session=requests.Session()
    session.headers['User-Agent']='ADAS-edge-offline/1.0 (github.com/wante30/adas-edge)'
    for row in json.loads(args.manifest.read_text(encoding='utf-8'))['images']:
        path=Path(row['path']);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():data=path.read_bytes()
        else:
            source=row['source'];info=source.get('info',{})
            url=source.get('download_url') or info.get('thumburl') or info['url']
            response=session.get(url,timeout=60);response.raise_for_status();data=response.content
            if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Downloaded hash mismatch '+row['id'])
            with path.open('xb') as stream:stream.write(data)
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Existing hash mismatch '+row['id'])
        print(row['id'],len(data))


if __name__=='__main__':main()
