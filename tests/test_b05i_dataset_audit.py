from pathlib import Path

import pytest

CLEAN = Path(r"D:\Datasets\adas_edge_training\b05i_clean_20261024\pseudo_A")


@pytest.mark.skipif(not CLEAN.exists(), reason="B05I local dataset is not present")
def test_clean_split_labels_are_nonempty_and_inside_image():
    labels = list((CLEAN / "labels").rglob("*.txt"))
    assert labels
    for path in labels:
        rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows, f"empty label leaked into main dataset: {path}"
        for row in rows:
            assert row[0] == "0"
            cx, cy, w, h = map(float, row[1:])
            assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
            assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


@pytest.mark.skipif(not CLEAN.exists(), reason="B05I local dataset is not present")
def test_pending_candidates_are_not_main_training_images():
    # A pending image must not appear as a main image with an empty label.
    main_images = {p.stem for p in (CLEAN / "images").rglob("*.jpg")}
    empty = [p for p in (CLEAN / "labels").rglob("*.txt")
             if p.stem in main_images and not p.read_text(encoding="utf-8").strip()]
    assert not empty
