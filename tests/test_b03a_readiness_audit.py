import json

import cv2
import numpy as np

from tools.audit_b03a_readiness import audit_dataset


def test_audit_counts_readable_images_and_duplicate_boxes(tmp_path) -> None:
    root = tmp_path / "dataset"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image = image_dir / "video-a-frame-001.jpg"
    assert cv2.imwrite(str(image), np.zeros((16, 32, 3), dtype=np.uint8))
    (label_dir / "video-a-frame-001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text("names:\n  0: traffic light\n", encoding="utf-8")

    result = audit_dataset(root, dataset_yaml)

    assert result["total_readable_images"] == 1
    assert result["classes"]["traffic light"] == {"instances": 2, "images": 1}
    assert result["exact_duplicate_boxes"] == 1


def test_audit_reads_bdd_traffic_light_color_attributes(tmp_path) -> None:
    root = tmp_path / "dataset"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text("names:\n  0: traffic light\n", encoding="utf-8")
    raw_json = tmp_path / "det_val.json"
    raw_json.write_text(
        json.dumps(
            [
                {
                    "labels": [
                        {
                            "category": "traffic light",
                            "attributes": {"trafficLightColor": "red"},
                        }
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )

    result = audit_dataset(root, dataset_yaml, raw_json)

    assert result["raw_json_traffic_light_color"] == {
        "traffic_light_labels": 1,
        "labels_with_trafficLightColor": 1,
        "values": {"red": 1},
    }
