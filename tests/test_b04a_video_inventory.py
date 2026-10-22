"""Tests for B-04A inventory: no unapproved video becomes a training proposal."""
import csv

from tools import b04a_video_inventory as inv


def test_sampling_is_sparse_and_bounded():
    assert inv.sample_times(12, 5, 10) == [2.5, 7.5]
    assert len(inv.sample_times(3600, 5, 5)) == 5


def test_official_test_never_train():
    row = {"source_kind": "official_test", "permission": "allowed",
           "license_note": "explicitly authorized", "split": "train"}
    assert inv.eligible(row)[0] is False


def test_rights_unknown_never_train():
    row = {"source_kind": "team_recorded", "permission": "unknown",
           "license_note": "", "split": "train"}
    assert inv.eligible(row)[0] is False


def test_inventory_preserves_video_level_split_and_no_frames(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"placeholder")
    registry = tmp_path / "registry.csv"
    with registry.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=inv.FIELDS)
        writer.writeheader()
        writer.writerow({"video_path": str(video), "source_id": "trip-001",
                         "source_kind": "team_recorded", "permission": "allowed",
                         "split": "train", "license_note": "rights reviewed"})
    monkeypatch.setattr(inv, "inspect_video", lambda path: (20.0, "ok"))
    report, proposals = inv.inventory(registry, gap_s=5, max_total=5)
    assert report["status"] == "PARTIAL"
    assert len(proposals) == 4
    assert all(row["annotation_status"] == "NOT_ANNOTATED" for row in proposals)
    assert all(row["training_ready"] == "false" for row in proposals)
    assert list(tmp_path.glob("*.jpg")) == []


def test_sequence_reuse_across_train_and_val_is_error(tmp_path, monkeypatch):
    for name in ("a.mp4", "b.mp4"):
        (tmp_path / name).write_bytes(b"x")
    registry = tmp_path / "registry.csv"
    with registry.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=inv.FIELDS)
        writer.writeheader()
        for name, split in (("a.mp4", "train"), ("b.mp4", "val")):
            writer.writerow({"video_path": str(tmp_path / name), "source_id": "one-drive",
                             "source_kind": "team_recorded", "permission": "allowed",
                             "split": split, "license_note": "reviewed"})
    monkeypatch.setattr(inv, "inspect_video", lambda path: (5.0, "ok"))
    report, _ = inv.inventory(registry)
    assert report["status"] == "FAIL"
    assert any("leakage" in error for error in report["errors"])
