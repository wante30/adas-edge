from competition_ai.model_profiles import BDD10, get_profile, validate_classes
from competition_ai.b05a_postprocess import (
    BDD_NAMES,
    bdd_class_name,
    keep_main_detection,
    keep_specialist_detection,
    normalize_ui_class,
)


def test_bdd_class_ids_are_not_coco_ids():
    assert bdd_class_name(0) == "pedestrian"
    assert bdd_class_name(3) == "truck"
    assert bdd_class_name(7) == "bicycle"
    assert bdd_class_name(8) == "traffic light"
    assert bdd_class_name(9) == "traffic sign"
    assert len(BDD_NAMES) == 10
    assert bdd_class_name(99) is None


def test_bdd_aliases_normalize_to_ui_vocabulary():
    assert normalize_ui_class("pedestrian") == "person"
    assert normalize_ui_class("rider") == "rider"
    assert normalize_ui_class("motor") == "motorcycle"
    assert normalize_ui_class("bike") == "bicycle"
    assert normalize_ui_class("traffic sign") is None


def test_main_model_rejects_traffic_lights_and_signs():
    assert keep_main_detection("traffic light", 0.99) is False
    assert keep_main_detection("traffic sign", 0.99) is False


def test_main_model_rejects_low_confidence_two_wheelers():
    assert keep_main_detection("motorcycle", 0.84) is False
    assert keep_main_detection("bicycle", 0.84) is False
    assert keep_main_detection("motorcycle", 0.85) is True
    assert keep_main_detection("bicycle", 0.86) is True


def test_main_model_keeps_vehicles_and_people():
    assert keep_main_detection("car", 0.25) is True
    assert keep_main_detection("truck", 0.25) is True
    assert keep_main_detection("bus", 0.25) is True
    assert keep_main_detection("pedestrian", 0.25) is True
    assert keep_main_detection("rider", 0.25) is True


def test_specialist_keeps_traffic_light_hypotheses():
    assert keep_specialist_detection(0.05) is True
    assert keep_specialist_detection(0.04) is False


def test_explicit_bdd_profile_is_complete_and_ordered():
    names = get_profile("bdd10")
    assert names == BDD10
    assert len(names) == 10
    assert names[3] == "truck"
    assert names[6] == "motorcycle"
    assert names[7] == "bicycle"
    assert names[8] == "traffic light"
    assert names[9] == "traffic sign"


def test_profile_validation_rejects_wrong_count():
    try:
        validate_classes(["car", "truck"], 10)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong class count must fail")
