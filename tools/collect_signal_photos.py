"""Download documented real photographs from Wikimedia Commons, with provenance.

No diagrams or generated images are used. Downloads are cached and not replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import requests

TITLES = {
    '00': 'Green light, red left turn arrow (41963560864).jpg',
    '01': 'UK Red Traffic Light With Green Filter Arrow - geograph.org.uk - 6409628.jpg',
    '02': 'Red arrow traffic lights.JPG',
    '03': 'U-turn traffic light.jpg',
    '04': 'UK Green Filter Traffic Light Signal Arrow - geograph.org.uk - 5094601.jpg',
    '08': 'US 62-PA 8 nb at Liberty Street, Franklin, June 2024.jpg',
    '09': 'PA 512 sb in Pen Argyl (1), Nov. 2022.jpg',
    '10': 'Traffic light at US98 and CR283.jpg',
    '11': 'Jianshe Avenue & Taibei Road, June 2023.jpg',
    '12': 'Traffic signal - red and right arrow, Otsu 2024-01.jpg',
    '13': 'Red and yellow traffic lights.jpg',
    '14': 'Traffic light profile.jpg',
    '15': 'North start of Niujie St (20201125145623).jpg',
    '16': 'Traffic light at Jeffrey Road and Smoketree.jpg',
    '17': 'UK Green Filter Arrow Traffic Light Signal Part 2 - geograph.org.uk - 5086756.jpg',
    '18': 'Tianjin Traffic Lights 2.jpg',
    '19': 'Signal LED.jpg',
    '20': 'Tamura Cross Road.jpg',
    '21': 'Flashing LT1.jpg',
    '22': 'New permissive left-turn signal (3423874873).jpg',
    '23': 'UK Green Traffic Light^Signal Filter Arrows Pointing Going Straight On And Turning Left - geograph.org.uk - 5113135.jpg',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='*', default=list(TITLES))
    parser.add_argument('--manifest', type=Path, help='Fetch exact recorded URLs and verify hashes')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    indexed = {entry['id']: entry for entry in manifest}
    recorded = {r['id']: r for r in json.loads(args.manifest.read_text(encoding='utf-8'))} if args.manifest else {}
    session = requests.Session()
    session.headers['User-Agent'] = 'ADAS-edge-research/1.0 (github.com/wante30/adas-edge)'
    for key in args.ids:
        if key in indexed and (args.output/(key+'.jpg')).exists():
            if key in recorded and hashlib.sha256((args.output/(key+'.jpg')).read_bytes()).hexdigest() != recorded[key]['sha256']:
                raise ValueError('Cached hash mismatch: '+key)
            continue
        title = TITLES[key]
        try:
            if key in recorded:
                previous = recorded[key]
                info = dict(url=previous['original_url'], descriptionurl=previous['page'], extmetadata=previous['metadata'])
                download_url = previous['download_url']
            else:
                response = session.get('https://commons.wikimedia.org/w/api.php', params={
                    'action': 'query', 'format': 'json', 'titles': 'File:'+title,
                    'prop': 'imageinfo', 'iiprop': 'url|extmetadata', 'iiurlwidth': 1920}, timeout=30)
                response.raise_for_status()
                info = next(iter(response.json()['query']['pages'].values()))['imageinfo'][0]
                download_url = info.get('thumburl', info['url'])
            response = session.get(download_url, timeout=60)
            response.raise_for_status()
            digest = hashlib.sha256(response.content).hexdigest()
            if key in recorded and digest != recorded[key]['sha256']:
                raise ValueError('Downloaded hash mismatch: '+key)
            path = args.output/(key+'.jpg')
            with path.open('xb') as output:
                output.write(response.content)
            indexed[key] = dict(id=key, title=title, page=info['descriptionurl'],
                original_url=info['url'], download_url=download_url,
                sha256=digest, metadata=info['extmetadata'])
            manifest_path.write_text(json.dumps(list(indexed.values()), ensure_ascii=False, indent=2), encoding='utf-8')
            print(key, len(response.content), flush=True)
        except (requests.RequestException, KeyError, FileExistsError) as error:
            print(key, str(error), flush=True)
        time.sleep(2)


if __name__ == '__main__':
    main()
