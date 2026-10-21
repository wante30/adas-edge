import json

from tools.prepare_bdd_road_subset import _records, _split_records


def test_pedestrian_maps_to_person_but_rider_is_not_merged(tmp_path) -> None:
    labels_path = tmp_path / "labels.json"
    box = {"x1": 1, "y1": 2, "x2": 11, "y2": 22}
    labels_path.write_text(
        json.dumps(
            [
                {
                    "name": "frame.jpg",
                    "labels": [
                        {"category": category, "box2d": box}
                        for category in ("person", "pedestrian", "rider")
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    records = _records(labels_path, set())

    assert [label[0] for label in records[0]["labels"]] == ["person", "person"]


def test_validation_split_keeps_bdd_video_sequences_together() -> None:
    records = [
        {"name": "video-a-frame-001.jpg"},
        {"name": "video-a-frame-002.jpg"},
        {"name": "video-b-frame-001.jpg"},
        {"name": "video-c-frame-001.jpg"},
    ]

    train, val = _split_records(records, val_fraction=0.25, seed=17)

    train_sequences = {record["name"].rsplit("-", 1)[0] for record in train}
    val_sequences = {record["name"].rsplit("-", 1)[0] for record in val}
    assert train and val
    assert train_sequences.isdisjoint(val_sequences)
