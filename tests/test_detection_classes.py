"""Tests for finalized common field-detection class vocabulary."""

from pathlib import Path
import yaml
import pytest

from idocr.data.annotation_schema import (
    DETECTOR_CLASSES,
    DOCUMENT_FIELDS,
    FIELD_NAME_TO_DETECTOR_CLASS,
    get_detector_class_id,
    get_detector_field_name,
    is_valid_field,
)
from idocr.types import DocumentType


def test_detector_class_ids_and_names_unique():
    class_ids = list(DETECTOR_CLASSES.keys())
    field_names = list(DETECTOR_CLASSES.values())

    # Check uniqueness
    assert len(class_ids) == len(set(class_ids)), "Detector class IDs must be unique"
    assert len(field_names) == len(set(field_names)), "Detector field names must be unique"

    # Check contiguous IDs 0..5
    assert sorted(class_ids) == list(range(6))


def test_expected_detector_mappings():
    expected = {
        0: "name",
        1: "date_of_birth",
        2: "gender",
        3: "aadhaar_number",
        4: "pan_number",
        5: "fathers_name",
    }
    assert DETECTOR_CLASSES == expected


def test_shared_fields_map_to_same_detector_class():
    # 'name' is shared across Aadhaar, PAN, and DL
    assert get_detector_class_id("name") == 0
    # 'date_of_birth' is shared across Aadhaar, PAN, and DL
    assert get_detector_class_id("date_of_birth") == 1

    # In Aadhaar, name is class 0 in detector
    # In PAN, name is class 0 in detector
    assert get_detector_field_name(0) == "name"
    assert get_detector_field_name(1) == "date_of_birth"


def test_document_specific_fields():
    assert get_detector_class_id("gender") == 2
    assert get_detector_class_id("aadhaar_number") == 3
    assert get_detector_class_id("pan_number") == 4
    assert get_detector_class_id("fathers_name") == 5


def test_class_4_not_in_detector_vocabulary():
    assert get_detector_class_id("class_4") is None
    assert get_detector_class_id("unknown") is None
    assert "class_4" not in FIELD_NAME_TO_DETECTOR_CLASS
    assert "class_4" not in DETECTOR_CLASSES.values()


def test_document_type_not_in_detector_vocabulary():
    assert get_detector_class_id("aadhaar") is None
    assert get_detector_class_id("pan") is None
    assert get_detector_class_id("driving_license") is None


def test_detection_classes_yaml_config():
    config_path = Path("configs/detection_classes.yaml")
    assert config_path.exists(), "configs/detection_classes.yaml must exist"

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes_cfg = cfg.get("classes", {})
    assert len(classes_cfg) == 6

    for cid, info in classes_cfg.items():
        expected_name = DETECTOR_CLASSES[cid]
        assert info["name"] == expected_name

    # Check source mappings
    source_map = cfg.get("source_to_detector_mapping", {})
    # Aadhaar: 0->3 (aadhaar_number), 1->1 (DOB), 2->2 (gender), 3->0 (name), 4->null
    assert source_map["aadhaar"][0] == 3
    assert source_map["aadhaar"][1] == 1
    assert source_map["aadhaar"][2] == 2
    assert source_map["aadhaar"][3] == 0
    assert source_map["aadhaar"][4] is None

    # PAN: 0->1 (DOB), 1->5 (father's name), 2->0 (name), 3->4 (pan_number)
    assert source_map["pan"][0] == 1
    assert source_map["pan"][1] == 5
    assert source_map["pan"][2] == 0
    assert source_map["pan"][3] == 4


def test_vocabulary_extensibility_for_driving_license():
    # Verify that future DL fields can be appended starting at ID 6 without modifying 0-5
    future_dl_fields = ["license_number", "issue_date", "expiry_date", "address", "vehicle_classes"]
    simulated_detector_vocab = dict(DETECTOR_CLASSES)
    for idx, field_name in enumerate(future_dl_fields, start=6):
        simulated_detector_vocab[idx] = field_name

    # Existing 0..5 IDs remain completely unchanged
    for cid in range(6):
        assert simulated_detector_vocab[cid] == DETECTOR_CLASSES[cid]

    assert len(simulated_detector_vocab) == 11
