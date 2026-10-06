"""Tests for unified annotation schema and document-specific field representations."""

import json
import pytest

from idocr.data.annotation_schema import (
    DOCUMENT_FIELDS,
    AnnotatedDocument,
    AnnotatedField,
    DocumentProvenance,
    FieldGeometry,
    FieldTranscription,
    FieldValidationRule,
    TranscriptionStatus,
    get_valid_fields,
    is_valid_field,
)
from idocr.types import BBox, DocumentType


def test_aadhaar_fields_representation():
    valid_fields = get_valid_fields(DocumentType.AADHAAR)
    assert valid_fields == {"aadhaar_number", "date_of_birth", "gender", "name"}

    fields = [
        AnnotatedField(
            field_name="aadhaar_number",
            geometry=FieldGeometry.from_yolo_box(0.5, 0.76, 0.36, 0.10),
            original_class_id=0,
        ),
        AnnotatedField(
            field_name="date_of_birth",
            geometry=FieldGeometry.from_yolo_box(0.62, 0.42, 0.15, 0.06),
            original_class_id=1,
        ),
        AnnotatedField(
            field_name="gender",
            geometry=FieldGeometry.from_yolo_box(0.44, 0.48, 0.10, 0.05),
            original_class_id=2,
        ),
        AnnotatedField(
            field_name="name",
            geometry=FieldGeometry.from_yolo_box(0.41, 0.33, 0.21, 0.07),
            original_class_id=3,
        ),
    ]

    doc = AnnotatedDocument(
        image_path="train/images/sample_aadhaar.jpg",
        document_type=DocumentType.AADHAAR,
        image_size=(640, 640),
        fields=fields,
        provenance=DocumentProvenance(
            source_dataset="aadhaar",
            source_split="train",
            source_group_id="aadhaar_src_0001",
            sha256="abc123",
        ),
    )

    assert len(doc.fields) == 4
    doc.validate_fields(strict=True)  # Should pass with no errors


def test_pan_fields_representation():
    valid_fields = get_valid_fields(DocumentType.PAN)
    assert valid_fields == {"name", "fathers_name", "date_of_birth", "pan_number"}

    fields = [
        AnnotatedField(
            field_name="date_of_birth",
            geometry=FieldGeometry.from_yolo_box(0.17, 0.55, 0.14, 0.04),
            original_class_id=0,
        ),
        AnnotatedField(
            field_name="fathers_name",
            geometry=FieldGeometry.from_yolo_box(0.25, 0.46, 0.24, 0.04),
            original_class_id=1,
        ),
        AnnotatedField(
            field_name="name",
            geometry=FieldGeometry.from_yolo_box(0.24, 0.39, 0.23, 0.04),
            original_class_id=2,
        ),
        AnnotatedField(
            field_name="pan_number",
            geometry=FieldGeometry.from_yolo_box(0.28, 0.52, 0.21, 0.04),
            original_class_id=3,
        ),
    ]

    doc = AnnotatedDocument(
        image_path="train/images/sample_pan.jpg",
        document_type=DocumentType.PAN,
        image_size=(2339, 1653),
        fields=fields,
        provenance=DocumentProvenance(
            source_dataset="pan",
            source_split="train",
            source_group_id="pan_src_0001",
        ),
    )

    assert len(doc.fields) == 4
    doc.validate_fields(strict=True)


def test_bbox_geometry_conversions():
    # Normalized YOLO box
    geom = FieldGeometry.from_yolo_box(cx=0.5, cy=0.5, w=0.2, h=0.1, is_normalized=True)
    assert geom.bbox.x1 == pytest.approx(0.4)
    assert geom.bbox.y1 == pytest.approx(0.45)
    assert geom.bbox.x2 == pytest.approx(0.6)
    assert geom.bbox.y2 == pytest.approx(0.55)
    assert geom.is_normalized is True

    # Convert to absolute pixel coordinates
    abs_geom = geom.to_absolute(width=1000, height=500)
    assert abs_geom.is_normalized is False
    assert abs_geom.bbox.x1 == pytest.approx(400)
    assert abs_geom.bbox.y1 == pytest.approx(225)
    assert abs_geom.bbox.x2 == pytest.approx(600)
    assert abs_geom.bbox.y2 == pytest.approx(275)

    # Convert back to normalized
    norm_geom = abs_geom.to_normalized(width=1000, height=500)
    assert norm_geom.is_normalized is True
    assert norm_geom.bbox.x1 == pytest.approx(0.4)


def test_polygon_geometry():
    # 5-point polygon coordinates: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
    raw_coords = [0.1, 0.2, 0.15, 0.6, 0.8, 0.7, 0.85, 0.3, 0.1, 0.2]
    geom = FieldGeometry.from_yolo_polygon(raw_coords, is_normalized=True)

    assert geom.polygon is not None
    assert len(geom.polygon) == 5
    assert geom.bbox.x1 == pytest.approx(0.1)
    assert geom.bbox.y1 == pytest.approx(0.2)
    assert geom.bbox.x2 == pytest.approx(0.85)
    assert geom.bbox.y2 == pytest.approx(0.7)


def test_missing_transcription_default():
    field = AnnotatedField(
        field_name="pan_number",
        geometry=FieldGeometry.from_yolo_box(0.5, 0.5, 0.2, 0.1),
    )
    assert field.transcription.text is None
    assert field.transcription.status == TranscriptionStatus.MISSING
    assert field.transcription.language is None


def test_populated_transcription():
    field = AnnotatedField(
        field_name="aadhaar_number",
        geometry=FieldGeometry.from_yolo_box(0.5, 0.8, 0.3, 0.05),
        transcription=FieldTranscription(
            text="1234 5678 9012",
            status=TranscriptionStatus.GROUND_TRUTH,
            language="en",
            script="latin",
            confidence=1.0,
        ),
        validation_rule=FieldValidationRule(
            expected_pattern=r"^\d{4}\s\d{4}\s\d{4}$",
            expected_type="numeric",
            checksum_algorithm="verhoeff",
        ),
    )
    assert field.transcription.text == "1234 5678 9012"
    assert field.transcription.status == TranscriptionStatus.GROUND_TRUTH
    assert field.validation_rule.checksum_algorithm == "verhoeff"


def test_class_4_rejected_as_semantic_field():
    assert is_valid_field(DocumentType.AADHAAR, "class_4") is False
    assert is_valid_field(DocumentType.AADHAAR, "unknown") is False
    assert is_valid_field(DocumentType.AADHAAR, "address") is False

    invalid_field = AnnotatedField(
        field_name="class_4",
        geometry=FieldGeometry.from_yolo_box(0.5, 0.5, 0.2, 0.1),
    )
    doc = AnnotatedDocument(
        image_path="test.jpg",
        document_type=DocumentType.AADHAAR,
        image_size=(640, 640),
        fields=[invalid_field],
    )

    with pytest.raises(ValueError, match="Field 'class_4' is not a valid semantic field"):
        doc.validate_fields(strict=True)


def test_serialization_roundtrip():
    field1 = AnnotatedField(
        field_name="name",
        geometry=FieldGeometry.from_yolo_box(0.4, 0.3, 0.2, 0.05),
        transcription=FieldTranscription(
            text="John Doe",
            status=TranscriptionStatus.PREDICTED,
            confidence=0.95,
        ),
        original_class_id=3,
    )
    doc = AnnotatedDocument(
        image_path="sample.jpg",
        document_type=DocumentType.AADHAAR,
        image_size=(640, 640),
        fields=[field1],
        provenance=DocumentProvenance(
            source_dataset="aadhaar",
            source_split="train",
            source_group_id="aadhaar_src_0010",
            sha256="deadbeef",
        ),
        metadata={"annotator": "synthetic"},
    )

    doc_dict = doc.to_dict()
    json_str = json.dumps(doc_dict)  # Verify JSON serializability
    loaded_dict = json.loads(json_str)

    restored_doc = AnnotatedDocument.from_dict(loaded_dict)
    assert restored_doc.image_path == "sample.jpg"
    assert restored_doc.document_type == DocumentType.AADHAAR
    assert len(restored_doc.fields) == 1
    assert restored_doc.fields[0].field_name == "name"
    assert restored_doc.fields[0].transcription.text == "John Doe"
    assert restored_doc.provenance.source_group_id == "aadhaar_src_0010"
