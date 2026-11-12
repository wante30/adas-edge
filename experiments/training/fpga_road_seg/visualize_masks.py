"""Create contact sheets from real converted BDD masks, including weak crosswalk curves."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = np.array([[0, 210, 100], [245, 80, 50], [0, 155, 255], [235, 60, 220]], np.float32)
NAMES = ['drivable', 'solid', 'dashed', 'crosswalk weak curve']


def sheet(manifest, output, count=12, mask_dir=None):
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line]
    if mask_dir is not None:
        mask_dir = Path(mask_dir)
        rows = [dict(row, masks=str(mask_dir / f'{i:08d}.npz'))
                for i, row in enumerate(rows) if (mask_dir / f'{i:08d}.npz').is_file()]
    weak = []
    other = []
    for row in rows:
        with np.load(row['masks'], allow_pickle=False) as masks:
            if masks['target'][3].sum():
                weak.append(row)
            else:
                other.append(row)
        if len(weak) >= count:
            break
    selected = (weak + other)[:count]
    cols, thumb_w, thumb_h, label_h = 3, 480, 270, 28
    canvas = Image.new('RGB', (cols * thumb_w, ((len(selected) + cols - 1) // cols) * (thumb_h + label_h)), 'white')
    draw = ImageDraw.Draw(canvas)
    weak_count = 0
    for i, row in enumerate(selected):
        with Image.open(row['image']) as im:
            image = im.convert('RGB').resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        with np.load(row['masks'], allow_pickle=False) as masks:
            target = masks['target']
            valid = masks['valid']
            weak_count += int(target[3].sum() > 0)
            display = np.asarray(image).astype(np.float32)
            for c in range(4):
                positive = Image.fromarray(target[c].astype(np.uint8) * 255).resize((thumb_w, thumb_h), Image.Resampling.NEAREST)
                active = np.asarray(positive) > 0
                display[active] = display[active] * 0.55 + COLORS[c] * 0.45
            y = Image.fromarray(display.clip(0, 255).astype(np.uint8))
        x, y0 = (i % cols) * thumb_w, (i // cols) * (thumb_h + label_h)
        canvas.paste(y, (x, y0 + label_h))
        weak_px = int(target[3].sum())
        valid_px = [int(v.sum()) for v in valid]
        draw.text((x + 4, y0 + 4), f"{row['id']} | cw weak px={weak_px} | valid={valid_px}", fill='black')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return dict(samples=len(selected), weak_curve_samples=weak_count, output=str(output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest')
    source.add_argument('--index', help='JSONL source index for inspecting an in-progress conversion')
    p.add_argument('--mask-dir', help='Required with --index; directory containing completed numbered NPZ masks')
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=12)
    a = p.parse_args()
    if a.index and not a.mask_dir:
        p.error('--mask-dir is required with --index')
    print(json.dumps(sheet(a.manifest or a.index, a.output, a.count, a.mask_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
