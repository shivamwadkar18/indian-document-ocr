"""Tests for YOLO to Unified Annotation Schema converter."""

from pathlib import Path
from PIL import Image
import pytest

from idocr.data.annotation_schema import (
    AnnotatedDocument,
    FieldGeometry,
    TranscriptionStatus,
)
from idocr.data.converters import (
    ConversionStats,
    YoloDatasetConverter,
    load_documents_jsonl,
    save_documents_jsonl,
)
from idocr.types import BBox, DocumentType


@pytest.fixture
def mock_dataset_tree(tmp_path: Path):
    """Create a small mock YOLO dataset tree with Aadhaar and PAN samples."""
    aadhaar_dir = tmp_path / "aadhaar"
    pan_dir = tmp_path / "pan"

    for d in [aadhaar_dir, pan_dir]:
        (d / "train" / "images").mkdir(parents=True)
        (d / "train" / "labels").mkdir(parents=True)

    # 1. Mock Aadhaar image & label
    img_a = Image.new("RGB", (640, 640), color=(200, 200, 200))
    img_a.save(aadhaar_dir / "train" / "images" / "aadh_001.jpg")

    # Label contains class 0, 1, 2, 3 and class 4 (box and polygon)
    label_a_lines = [
        "0 0.50 0.76 0.36 0.10",   # Aadhaar number
        "1 0.62 0.42 0.15 0.06",   # DOB
        "2 0.44 0.48 0.10 0.05",   # Gender
        "3 0.41 0.33 0.21 0.07",   # Name
        "4 0.50 0.50 0.40 0.20",   # Class 4 box (noise)
        "4 0.10 0.10 0.20 0.10 0.20 0.30 0.10 0.30", # Class 4 polygon (noise)
    ]
    (aadhaar_dir / "train" / "labels" / "aadh_001.txt").write_text("\n".join(label_a_lines), encoding="utf-8")

    # 2. Mock PAN image & label
    img_p = Image.new("RGB", (1000, 500), color=(220, 220, 220))
    img_p.save(pan_dir / "train" / "images" / "pan_001.jpg")

    label_p_lines = [
        "0 0.17 0.55 0.14 0.04",   # DOB
        "1 0.25 0.46 0.24 0.04",   # Father's name
        "2 0.24 0.39 0.23 0.04",   # Name
        "3 0.28 0.52 0.21 0.04",   # PAN number
    ]
    (pan_dir / "train" / "labels" / "pan_001.txt").write_text("\n".join(label_p_lines), encoding="utf-8")

    return {"aadhaar": aadhaar_dir, "pan": pan_dir}


def test_aadhaar_yolo_conversion(mock_dataset_tree):
    conv = YoloDatasetConverter(document_type=DocumentType.AADHAAR)
    img_path = mock_dataset_tree["aadhaar"] / "train" / "images" / "aadh_001.jpg"

    doc, total_anns, excluded_c4 = conv.convert_file(img_path)

    assert total_anns == 6
    assert excluded_c4 == 2  # 1 box + 1 polygon of class 4
    assert len(doc.fields) == 4

    field_map = {f.field_name: f for f in doc.fields}
    assert set(field_map.keys()) == {"aadhaar_number", "date_of_birth", "gender", "name"}

    # Verify original class IDs
    assert field_map["aadhaar_number"].original_class_id == 0
    assert field_map["date_of_birth"].original_class_id == 1
    assert field_map["gender"].original_class_id == 2
    assert field_map["name"].original_class_id == 3

    # Verify OCR transcriptions are unpopulated defaults
    for f in doc.fields:
        assert f.transcription.text is None
        assert f.transcription.status == TranscriptionStatus.MISSING


def test_pan_yolo_conversion(mock_dataset_tree):
    conv = YoloDatasetConverter(document_type=DocumentType.PAN)
    img_path = mock_dataset_tree["pan"] / "train" / "images" / "pan_001.jpg"

    doc, total_anns, excluded_c4 = conv.convert_file(img_path)

    assert total_anns == 4
    assert excluded_c4 == 0
    assert len(doc.fields) == 4

    field_map = {f.field_name: f for f in doc.fields}
    assert set(field_map.keys()) == {"date_of_birth", "fathers_name", "name", "pan_number"}

    # Verify original class IDs
    assert field_map["date_of_birth"].original_class_id == 0
    assert field_map["fathers_name"].original_class_id == 1
    assert field_map["name"].original_class_id == 2
    assert field_map["pan_number"].original_class_id == 3


def test_geometry_coordinates_precision(mock_dataset_tree):
    conv = YoloDatasetConverter(document_type=DocumentType.PAN)
    img_path = mock_dataset_tree["pan"] / "train" / "images" / "pan_001.jpg"

    doc, _, _ = conv.convert_file(img_path)
    dob_field = next(f for f in doc.fields if f.field_name == "date_of_birth")

    # YOLO coords: cx=0.17, cy=0.55, w=0.14, h=0.04
    # Image size: 1000 x 500
    # Normalized: x1 = 0.17 - 0.07 = 0.10, x2 = 0.24, y1 = 0.55 - 0.02 = 0.53, y2 = 0.57
    assert dob_field.geometry.bbox.x1 == pytest.approx(0.10)
    assert dob_field.geometry.bbox.x2 == pytest.approx(0.24)
    assert dob_field.geometry.bbox.y1 == pytest.approx(0.53)
    assert dob_field.geometry.bbox.y2 == pytest.approx(0.57)

    # Convert to absolute pixel space
    abs_geom = dob_field.geometry.to_absolute(doc.image_size[0], doc.image_size[1])
    assert abs_geom.bbox.x1 == pytest.approx(100.0)
    assert abs_geom.bbox.x2 == pytest.approx(240.0)
    assert abs_geom.bbox.y1 == pytest.approx(265.0)
    assert abs_geom.bbox.y2 == pytest.approx(285.0)


def test_jsonl_serialization_roundtrip(tmp_path, mock_dataset_tree):
    conv = YoloDatasetConverter(document_type=DocumentType.AADHAAR)
    img_path = mock_dataset_tree["aadhaar"] / "train" / "images" / "aadh_001.jpg"
    doc, _, _ = conv.convert_file(img_path)

    jsonl_path = tmp_path / "documents.jsonl"
    save_documents_jsonl([doc], jsonl_path)

    loaded_docs = load_documents_jsonl(jsonl_path)
    assert len(loaded_docs) == 1
    loaded = loaded_docs[0]
    assert loaded.document_type == DocumentType.AADHAAR
    assert len(loaded.fields) == 4
    assert loaded.image_size == (640, 640)


@pytest.mark.skipif(not Path("data/processed/aadhaar").exists(), reason="Processed Aadhaar dataset missing")
def test_real_processed_aadhaar_conversion():
    aadhaar_dir = Path("data/processed/aadhaar")
    manifest_path = Path("data/processed/manifests/aadhaar_source_groups.csv")

    conv = YoloDatasetConverter(document_type=DocumentType.AADHAAR, manifest_path=manifest_path)
    docs, stats = conv.convert_dataset(aadhaar_dir)

    assert stats.input_images == 2646
    assert stats.converted_documents == 2646
    assert stats.input_annotations == 10310
    assert stats.included_fields == 10223
    assert stats.excluded_class_4_annotations == 87
    assert len(stats.conversion_errors) == 0


@pytest.mark.skipif(not Path("data/processed/pan").exists(), reason="Processed PAN dataset missing")
def test_real_processed_pan_conversion():
    pan_dir = Path("data/processed/pan")
    manifest_path = Path("data/processed/manifests/pan_source_groups.csv")

    conv = YoloDatasetConverter(document_type=DocumentType.PAN, manifest_path=manifest_path)
    docs, stats = conv.convert_dataset(pan_dir)

    assert stats.input_images == 1726
    assert stats.converted_documents == 1726
    assert stats.input_annotations == 6879
    assert stats.included_fields == 6879
    assert stats.excluded_class_4_annotations == 0
    assert len(stats.conversion_errors) == 0
