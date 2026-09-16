"""Identify which camera each finedet/nuimages parquet shard contains.

Reads only the footer plus the first row group of the ``image`` column via HTTP
range requests, so a 27-shard / 11 GB dataset is probed with a few hundred MB.
"""
from __future__ import annotations

import collections
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/finedet/nuimages/resolve/main/data/train-{index:05d}-of-00027.parquet"
CAMERA_RE = re.compile(r"__(CAM_[A-Z]+)__")
HEADERS = {"User-Agent": "Mozilla/5.0"}


class HttpRangeReader(io.RawIOBase):
    """Minimal seekable file object backed by HTTP range requests."""

    def __init__(self, url: str, timeout: int = 60) -> None:
        self.url = url
        self.timeout = timeout
        self.pos = 0
        request = urllib.request.Request(url, method="HEAD", headers=HEADERS)
        self.size = int(urllib.request.urlopen(request, timeout=timeout).headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int | None = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.pos
        if size == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + size, self.size) - 1
        request = urllib.request.Request(self.url, headers={**HEADERS, "Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(request, timeout=self.timeout).read()
        self.pos += len(data)
        return data


def camera_of(path: str) -> str:
    match = CAMERA_RE.search(path or "")
    return match.group(1) if match else "UNKNOWN"


def probe(shard: int) -> tuple[str, dict[str, int], int]:
    parquet = pq.ParquetFile(HttpRangeReader(BASE.format(index=shard)))
    table = parquet.read_row_group(0, columns=["image"])
    cameras = collections.Counter(camera_of(row.get("path") or "") for row in table.column("image").to_pylist())
    return cameras.most_common(1)[0][0], dict(cameras), parquet.metadata.num_rows


def main() -> None:
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("nuimages_shard_cameras.json")
    result: dict[str, str] = {}
    for shard in range(27):
        try:
            camera, detail, rows = probe(shard)
        except Exception as exc:  # network flakiness must not abort the sweep
            print(f"shard {shard:<3} ERR {type(exc).__name__}: {str(exc)[:110]}", flush=True)
            continue
        result[str(shard)] = camera
        print(f"shard {shard:<3} rows={rows:<6} {camera:<12} {detail}", flush=True)
    front = [int(k) for k, v in result.items() if v == "CAM_FRONT"]
    print(f"\nCAM_FRONT shards: {sorted(front)} -> ~{len(front) * 2500} images", flush=True)
    destination.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
