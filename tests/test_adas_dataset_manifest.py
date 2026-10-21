"""Contract checks for the independently developed ADAS dataset manifest auditor."""
from tools.audit_adas_dataset_manifest import DEFAULT_CLASSES, audit_rows

ALL = "|".join(DEFAULT_CLASSES)


def row(name, split="train", sequence="seq1", exhaustive=ALL):
    return {
        "sample_id": name,
        "dataset": "bdd100k",
        "sequence_id": sequence,
        "split": split,
        "classes_exhaustive": exhaustive,
        "license": "verified-by-operator",
        "source_uri": "local://a",
    }


def test_complete_separate_sequences_pass():
    report = audit_rows([
        row("a", "train", "trip1"),
        row("b", "val", "trip2"),
        row("c", "test", "trip3"),
    ])
    assert report["status"] == "PASS"
    assert report["counts"]["train"] == 1


def test_missing_traffic_lights_fails():
    partial = "|".join(c for c in DEFAULT_CLASSES if not c.endswith("_light") and "light_" not in c)
    report = audit_rows([row("a", exhaustive=partial)])
    assert report["status"] == "FAIL"
    assert "red_light" in "\n".join(report["errors"])


def test_train_test_same_sequence_is_leakage():
    report = audit_rows([row("a", "train", "trip1"), row("b", "test", "trip1")])
    assert report["status"] == "FAIL"
    assert any("sequence leakage" in msg for msg in report["errors"])


def test_duplicate_sample_rejected():
    report = audit_rows([row("a", "train", "trip1"), row("a", "train", "trip1")])
    assert report["status"] == "FAIL"
    assert any("duplicate" in msg for msg in report["errors"])


def test_missing_license_rejected():
    candidate = row("a")
    candidate["license"] = ""
    report = audit_rows([candidate])
    assert report["status"] == "FAIL"
    assert any("license" in msg for msg in report["errors"])


def test_empty_manifest_rejected():
    assert audit_rows([])["status"] == "FAIL"
