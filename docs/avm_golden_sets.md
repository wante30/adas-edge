# AVM golden sets

Five frame IDs are frozen: 0, 24, 49, 74, 99. The generated directories are `frame_000`, `frame_024`, `frame_049`, `frame_074`, and `frame_099`; each contains float RGB888, fixed RGB565, and diff images.

The four original A2D2 folders have `timestamp_alignment.csv` evidence. The added front-left/front-right folders use the same source-index sequence but have no per-frame timestamp sidecars, so six-camera timestamp skew is UNKNOWN and must be revalidated before hardware claims.
