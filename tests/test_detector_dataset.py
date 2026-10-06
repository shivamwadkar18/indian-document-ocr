"""Unit and integration tests for common field-detector dataset generation."""

from pathlib import Path
import pytest
import yaml

from idocr.data.annotation_schema import (
    DETECTOR_CLASSES,
    FIELD_NAME_TO_DETECTOR_CLASS,
    get_detector_class_id,
    get_detector_field_name,
)
from idocr.data.detection.builder import DetectorDatasetBuilder, DetectorDatasetStats
from idocr.data.detection.validation import DetectorDatasetValidator, DetectorValidationResult


@pytest.fixture
def dummy_dataset_env(tmp_path: Path):
    """Creates a miniature dummy processed dataset structure for unit tests."""
    aadhaar_dir = tmp_path / "processed" / "aadhaar"
    pan_dir = tmp_path / "processed" / "pan"
    out_dir = tmp_path / "processed" / "detector"
    manifests_dir = tmp_path / "processed" / "manifests"
    raw_dir = tmp_path / "raw"

    for split in ("train", "valid", "test"):
        (aadhaar_dir / split / "images").mkdir(parents=True)
        (aadhaar_dir / split / "labels").mkdir(parents=True)
        (pan_dir / split / "images").mkdir(parents=True)
        (pan_dir / split / "labels").mkdir(parents=True)

    manifests_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    (raw_dir / "aadhaar").mkdir()
    (raw_dir / "pan").mkdir()

    # Create Aadhaar sample in train:
    # img1: class 3 (name -> 0), class 0 (aadhaar_number -> 3), class 1 (dob -> 1), class 2 (gender -> 2), class 4 (noise -> excluded)
    (aadhaar_dir / "train" / "images" / "aadh_001.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    (aadhaar_dir / "train" / "labels" / "aadh_001.txt").write_text(
        "3 0.5 0.2 0.4 0.1\n0 0.5 0.8 0.6 0.1\n1 0.3 0.4 0.2 0.05\n2 0.7 0.4 0.1 0.05\n4 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    # img2: only class 4 (noise -> should produce empty label file)
    (aadhaar_dir / "train" / "images" / "aadh_only_4.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    (aadhaar_dir / "train" / "labels" / "aadh_only_4.txt").write_text("4 0.1 0.1 0.2 0.2\n", encoding="utf-8")

    # Create PAN sample in train:
    # pan1: class 2 (name -> 0), class 0 (dob -> 1), class 1 (fathers_name -> 5), class 3 (pan_number -> 4)
    (pan_dir / "train" / "images" / "pan_001.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    (pan_dir / "train" / "labels" / "pan_001.txt").write_text(
        "2 0.4 0.3 0.3 0.08\n0 0.4 0.5 0.2 0.06\n1 0.4 0.4 0.3 0.08\n3 0.7 0.8 0.4 0.1\n",
        encoding="utf-8",
    )

    # Valid split sample
    (aadhaar_dir / "valid" / "images" / "aadh_val.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    (aadhaar_dir / "valid" / "labels" / "aadh_val.txt").write_text("3 0.5 0.2 0.3 0.1\n", encoding="utf-8")
    (pan_dir / "valid" / "images" / "pan_val.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    (pan_dir / "valid" / "labels" / "pan_val.txt").write_text("3 0.6 0.7 0.3 0.1\n", encoding="utf-8")

    # Test split sample
    (aadhaar_dir / "test" / "images" / "aadh_test.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    (aadhaar_dir / "test" / "labels" / "aadh_test.txt").write_text("0 0.5 0.8 0.5 0.1\n", encoding="utf-8")
    (pan_dir / "test" / "images" / "pan_test.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    (pan_dir / "test" / "labels" / "pan_test.txt").write_text("2 0.5 0.3 0.4 0.1\n", encoding="utf-8")

    return {
        "aadhaar_dir": aadhaar_dir,
        "pan_dir": pan_dir,
        "out_dir": out_dir,
        "manifests_dir": manifests_dir,
        "raw_dir": raw_dir,
        "tmp_path": tmp_path,
    }


def test_aadhaar_and_pan_field_mappings():
    """1 & 2: Test field to detector class mapping for Aadhaar and PAN."""
    # Aadhaar mapping
    assert get_detector_class_id("name") == 0
    assert get_detector_class_id("date_of_birth") == 1
    assert get_detector_class_id("gender") == 2
    assert get_detector_class_id("aadhaar_number") == 3

    # PAN mapping
    assert get_detector_class_id("pan_number") == 4
    assert get_detector_class_id("fathers_name") == 5


def test_builder_class_4_exclusion(dummy_dataset_env):
    """3 & 4: Aadhaar Class 4 is excluded and produces empty label file when sole annotation."""
    env = dummy_dataset_env
    builder = DetectorDatasetBuilder(output_dir=env["out_dir"])
    stats = builder.build_dataset(
        aadhaar_processed_dir=env["aadhaar_dir"],
        pan_processed_dir=env["pan_dir"],
    )

    assert stats.excluded_class_4_count == 2
    assert stats.empty_label_images == 1

    # Check aadh_001.txt labels
    lbl_aadh_001 = (env["out_dir"] / "train" / "labels" / "aadh_001.txt").read_text()
    lines_001 = [l.strip() for l in lbl_aadh_001.splitlines() if l.strip()]
    # 4 valid targets (name->0, aadhaar_number->3, dob->1, gender->2), class 4 excluded
    assert len(lines_001) == 4
    class_ids_001 = {int(l.split()[0]) for l in lines_001}
    assert class_ids_001 == {0, 1, 2, 3}
    assert 4 not in class_ids_001

    # Check aadh_only_4.txt label: should exist and be empty
    lbl_only_4_path = env["out_dir"] / "train" / "labels" / "aadh_only_4.txt"
    assert lbl_only_4_path.exists()
    assert lbl_only_4_path.read_text().strip() == ""


def test_multiple_valid_fields_and_pan_mapping(dummy_dataset_env):
    """5 & 6: Test multiple valid fields and normalized YOLO coordinates."""
    env = dummy_dataset_env
    builder = DetectorDatasetBuilder(output_dir=env["out_dir"])
    builder.build_dataset(
        aadhaar_processed_dir=env["aadhaar_dir"],
        pan_processed_dir=env["pan_dir"],
    )

    lbl_pan_001 = (env["out_dir"] / "train" / "labels" / "pan_001.txt").read_text()
    lines_pan = [l.strip() for l in lbl_pan_001.splitlines() if l.strip()]
    assert len(lines_pan) == 4

    # PAN mapping: class 2->0 (name), class 0->1 (dob), class 1->5 (fathers_name), class 3->4 (pan_number)
    class_ids_pan = [int(l.split()[0]) for l in lines_pan]
    assert class_ids_pan == [0, 1, 5, 4]

    for line in lines_pan:
        parts = line.split()
        assert len(parts) == 5
        cid, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        assert 0 <= cid <= 5
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 < w <= 1.0
        assert 0.0 < h <= 1.0


def test_invalid_geometry_rejection(tmp_path: Path):
    """7: Rejects out of bounds coordinates."""
    bad_dir = tmp_path / "bad"
    (bad_dir / "train" / "images").mkdir(parents=True)
    (bad_dir / "train" / "labels").mkdir(parents=True)
    out_dir = tmp_path / "detector_out"

    (bad_dir / "train" / "images" / "bad.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    # cx = 1.5 (out of bounds)
    (bad_dir / "train" / "labels" / "bad.txt").write_text("3 1.5 0.5 0.2 0.2\n")

    builder = DetectorDatasetBuilder(output_dir=out_dir)
    stats = builder.build_dataset(aadhaar_processed_dir=bad_dir, pan_processed_dir=tmp_path / "nonexistent")

    assert len(stats.validation_errors) > 0
    assert "Out of bounds" in stats.validation_errors[0]


def test_unsupported_field_rejection(tmp_path: Path):
    """8: Rejects unmapped class ID."""
    bad_dir = tmp_path / "unmapped"
    (bad_dir / "train" / "images").mkdir(parents=True)
    (bad_dir / "train" / "labels").mkdir(parents=True)
    out_dir = tmp_path / "detector_out"

    (bad_dir / "train" / "images" / "unmapped.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    # class 99 is unmapped
    (bad_dir / "train" / "labels" / "unmapped.txt").write_text("99 0.5 0.5 0.2 0.2\n")

    builder = DetectorDatasetBuilder(output_dir=out_dir)
    stats = builder.build_dataset(aadhaar_processed_dir=bad_dir, pan_processed_dir=tmp_path / "nonexistent")

    assert len(stats.validation_errors) > 0
    assert "Unmapped source class 99" in stats.validation_errors[0]


def test_train_valid_test_preservation_and_pairing(dummy_dataset_env):
    """9, 10, 11: Preserves split distribution, image/label pairing, and no leakage."""
    env = dummy_dataset_env
    builder = DetectorDatasetBuilder(output_dir=env["out_dir"])
    stats = builder.build_dataset(
        aadhaar_processed_dir=env["aadhaar_dir"],
        pan_processed_dir=env["pan_dir"],
    )

    validator = DetectorDatasetValidator(
        dataset_dir=env["out_dir"],
        manifests_dir=env["manifests_dir"],
        raw_dir=env["raw_dir"],
        processed_dir=env["tmp_path"] / "processed",
    )
    res = validator.validate()

    assert res.is_valid
    assert res.pairing_passed
    assert res.classes_valid
    assert res.geometry_valid
    assert res.split_leakage_free
    assert res.images_by_split["train"] == 3
    assert res.images_by_split["valid"] == 2
    assert res.images_by_split["test"] == 2
    assert res.total_images == 7


def test_data_yaml_structure(dummy_dataset_env):
    """Validates data.yaml structure created for YOLO."""
    env = dummy_dataset_env
    builder = DetectorDatasetBuilder(output_dir=env["out_dir"])
    builder.build_dataset(
        aadhaar_processed_dir=env["aadhaar_dir"],
        pan_processed_dir=env["pan_dir"],
    )

    yaml_file = env["out_dir"] / "data.yaml"
    assert yaml_file.exists()

    with yaml_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["train"] == "train/images"
    assert data["val"] == "valid/images"
    assert data["test"] == "test/images"
    assert data["names"] == {
        0: "name",
        1: "date_of_birth",
        2: "gender",
        3: "aadhaar_number",
        4: "pan_number",
        5: "fathers_name",
    }


def test_real_processed_detector_dataset_integrity():
    """12 & 13: Full integration validation on the actual generated detector dataset."""
    detector_dir = Path("data/processed/detector")
    if not detector_dir.exists():
        pytest.skip("Detector dataset has not been generated yet.")

    validator = DetectorDatasetValidator(
        dataset_dir=detector_dir,
        manifests_dir=Path("data/processed/manifests"),
        raw_dir=Path("data/raw"),
        processed_dir=Path("data/processed"),
    )
    res = validator.validate()

    assert res.is_valid, f"Validation errors: {res.errors}"
    assert res.pairing_passed, "Every image must have a corresponding label file"
    assert res.classes_valid, "Class IDs must be restricted strictly to 0..5"
    assert res.geometry_valid, "All YOLO coordinates must be valid [0, 1] normalized boxes"
    assert res.split_leakage_free, "No cross-split source group leakage allowed"
    assert res.source_integrity_passed, "Raw and processed datasets must be intact"

    # Verify expected accounting
    assert res.total_images == 4372
    assert res.images_by_split == {"train": 3060, "valid": 654, "test": 658}
    assert res.total_targets == 17102
    assert set(res.targets_by_class.keys()) == set(range(6))
