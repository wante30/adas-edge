from pathlib import Path

import cv2
import numpy as np

from tools.evaluate_b03d_accuracy import build_ground_truth


def test_build_ground_truth_reads_detection_2020_voc(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    name = "frame-001.jpg"
    assert cv2.imwrite(str(images / name), np.zeros((720, 1280, 3), dtype=np.uint8))
    (labels / "frame-001.xml").write_text(
        """<annotation>
        <object><name>car</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>21</xmax><ymax>22</ymax></bndbox></object>
        <object><name>pedestrian</name><bndbox><xmin>5</xmin><ymin>6</ymin><xmax>15</xmax><ymax>26</ymax></bndbox></object>
        <object><name>rider</name><bndbox><xmin>30</xmin><ymin>30</ymin><xmax>40</xmax><ymax>40</ymax></bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )

    ground_truth, stats = build_ground_truth(labels, [name], images)

    assert stats["images_with_labels"] == 1
    assert stats["label_formats"] == {"voc_xml": 1}
    assert stats["per_class_instances"]["car"] == 1
    assert stats["per_class_instances"]["person"] == 1
    assert stats["skipped_category_counts"]["rider"] == 1
    assert len(ground_truth["annotations"]) == 2


def test_build_ground_truth_preserves_json_traffic_light_color(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    name = "frame-002.jpg"
    assert cv2.imwrite(str(images / name), np.zeros((720, 1280, 3), dtype=np.uint8))
    (labels / "frame-002.json").write_text(
        '{"frames":[{"objects":[{"category":"traffic light","attributes":{"trafficLightColor":"red"},'
        '"box2d":{"x1":1,"y1":2,"x2":11,"y2":12}}]}]}',
        encoding="utf-8",
    )

    _, stats = build_ground_truth(labels, [name], images)

    assert stats["traffic_light_color_counts"] == {"red": 1}
