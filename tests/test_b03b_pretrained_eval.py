from tools.b03b_pretrained_eval import canonicalize_model_names


def test_bdd_model_aliases_are_mapped_by_name_and_rider_stays_separate():
    names = {
        0: "person",
        1: "rider",
        2: "motor",
        3: "bike",
        4: "traffic light",
    }

    assert canonicalize_model_names(names) == {
        0: "person",
        1: "rider",
        2: "motorcycle",
        3: "bicycle",
        4: "traffic light",
    }
