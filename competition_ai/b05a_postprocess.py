"""B-05A class-name and false-positive post-processing for BDD YOLOv8n.

The RKNN BDD model emits ten BDD100K class ids.  The production console used
to interpret those ids with COCO_NAMES, which turned truck (id 3) into
motorcycle and traffic sign (id 9) into traffic light.  This module keeps the
model/runtime split explicit and makes the conservative filters testable.
"""
from __future__ import annotations

BDD_NAMES = (
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
)

# Classes that the competition UI can render; BDD aliases are normalized
# first, then this set is checked.
COMPETITION_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign",
}

# The BDD model's AP50 for these two classes is 0.0 in the B-03E holdout;
# requiring a high confidence keeps obvious false positives off the HUD.
WEAK_TWO_WHEELER_CONFIDENCE = 0.85


def bdd_class_name(class_id: int) -> str | None:
    """Return the BDD100K name for a model class id, or None for invalid ids."""
    if 0 <= int(class_id) < len(BDD_NAMES):
        return BDD_NAMES[int(class_id)]
    return None


def normalize_ui_class(name: str) -> str | None:
    """Map BDD names to the small UI vocabulary; None means do not draw."""
    if name in ("pedestrian", "rider"):
        return "person"
    if name in ("motor", "motorcycle"):
        return "motorcycle"
    if name in ("bike", "bicycle"):
        return "bicycle"
    if name == "traffic sign":
        return None
    return name


def keep_main_detection(name: str, confidence: float) -> bool:
    """Apply B-05A main-model filtering.

    Traffic lights belong to the dedicated Specialist.  The main BDD model's
    motor/bicycle AP is zero/low and it produced high-confidence truck->motor
    false boxes, so only high-confidence two-wheelers are admitted.
    """
    ui_name = normalize_ui_class(name)
    if ui_name is None:
        return False
    if ui_name == "traffic light":
        return False
    if ui_name in ("motorcycle", "bicycle") and confidence < WEAK_TWO_WHEELER_CONFIDENCE:
        return False
    return ui_name in COMPETITION_CLASSES


def keep_specialist_detection(confidence: float) -> bool:
    """Specialist detections are traffic-light hypotheses; keep its boxes."""
    return float(confidence) >= 0.05
