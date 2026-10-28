# AVM bandwidth and LUT budget

This is a theoretical external-memory traffic budget at the frozen working resolution. It counts selected RGB565 camera reads, one packed LUT read per output pixel, and one RGB565 output write; it is not a measured DDR trace.

| channels | input/frame | LUT/frame | output/frame | total/frame | 30 FPS | 60 FPS |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1638400 | 3686400 | 614400 | 5939200 | 178.176 MB/s | 356.352 MB/s |
| 6 | 2457600 | 3686400 | 614400 | 6758400 | 202.752 MB/s | 405.504 MB/s |

- One 96-bit entry is 12 bytes; 307200 entries consume 3686400 bytes (3.516 MiB).
- Valid and blend masks are separate 1-byte/pixel audit artifacts; the packed LUT already carries both flags.
