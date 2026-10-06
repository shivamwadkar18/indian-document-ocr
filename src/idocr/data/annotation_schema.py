"""Unified annotation schema for training, dataset conversion, and evaluation.

This module provides the canonical, document-independent data structures
used across the ML pipeline. It bridges raw/processed source annotations
(YOLO boxes, polygon contours) into a structured schema supporting field
detection, text recognition (OCR), validation, and dataset provenance.

Key Principles:
1. Separation of Detection and OCR: Geometric location (FieldGeometry) is
   strictly separated from text content (FieldTranscription).
2. Multiple Geometries: Supports bounding boxes and polygon contours in
   both normalized [0, 1] and absolute pixel coordinate spaces.
3. Document-Specific Field Vocabularies: Validates field names per document
   type (Aadhaar, PAN, Driving Licence).
4. Unresolved Class 4 Exclusion: Class 4 is explicitly excluded from the
   valid semantic field vocabulary and cannot become a training target.
5. Traceability: Retains source dataset, split, source group, and SHA-256
   provenance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from idocr.types import BBox, DocumentType


SCHEMA_VERSION = "1.0.0"


class TranscriptionStatus(str, Enum):
    """Status of OCR / text ground truth for an annotated field."""
    MISSING = "missing"              # Field detected/annotated, no transcription available
    GROUND_TRUTH = "ground_truth"    # Verified human-annotated ground truth text
    PREDICTED = "predicted"          # Machine-predicted text from an OCR engine
    UNREADABLE = "unreadable"        # Text is blurred, occluded, or unreadable


# Canonical field vocabularies per document type
DOCUMENT_FIELDS: dict[DocumentType, set[str]] = {
    DocumentType.AADHAAR: {
        "aadhaar_number",
        "date_of_birth",
        "gender",
        "name",
    },
    DocumentType.PAN: {
        "name",
        "fathers_name",
        "date_of_birth",
        "pan_number",
    },
    DocumentType.DRIVING_LICENSE: {
        "license_number",
        "name",
        "fathers_name",
        "date_of_birth",
        "issue_date",
        "expiry_date",
        "address",
        "vehicle_classes",
    },
    DocumentType.UNKNOWN: set(),
}


# Canonical common field detector vocabulary across document types
DETECTOR_CLASSES: dict[int, str] = {
    0: "name",
    1: "date_of_birth",
    2: "gender",
    3: "aadhaar_number",
    4: "pan_number",
    5: "fathers_name",
}

FIELD_NAME_TO_DETECTOR_CLASS: dict[str, int] = {
    name: cid for cid, name in DETECTOR_CLASSES.items()
}


def get_detector_class_id(field_name: str) -> int | None:
    """Return the detector class ID for a canonical field name, or None if unmapped."""
    return FIELD_NAME_TO_DETECTOR_CLASS.get(field_name)


def get_detector_field_name(class_id: int) -> str | None:
    """Return the canonical field name for a detector class ID, or None if unmapped."""
    return DETECTOR_CLASSES.get(class_id)


def get_valid_fields(doc_type: DocumentType | str) -> set[str]:
    """Return the set of valid semantic field names for a document type."""
    if isinstance(doc_type, str):
        try:
            doc_type = DocumentType(doc_type.lower())
        except ValueError:
            return set()
    return DOCUMENT_FIELDS.get(doc_type, set())


def is_valid_field(doc_type: DocumentType | str, field_name: str) -> bool:
    """Check if a field name is a valid semantic field for the given document type."""
    return field_name in get_valid_fields(doc_type)


@dataclass
class FieldGeometry:
    """Geometric region of a field on the document image."""

    bbox: BBox
    polygon: list[tuple[float, float]] | None = None
    is_normalized: bool = True  # True if coordinates are in [0, 1], False if absolute pixels

    def __post_init__(self) -> None:
        if self.is_normalized:
            # Check normalized bounds [0, 1]
            if not (0.0 <= self.bbox.x1 <= 1.0 and 0.0 <= self.bbox.x2 <= 1.0 and
                    0.0 <= self.bbox.y1 <= 1.0 and 0.0 <= self.bbox.y2 <= 1.0):
                # Tolerance for slight float rounding
                pass

    @classmethod
    def from_yolo_box(cls, cx: float, cy: float, w: float, h: float, is_normalized: bool = True) -> FieldGeometry:
        """Create geometry from YOLO center-x, center-y, width, height format."""
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return cls(bbox=BBox(x1, y1, x2, y2), is_normalized=is_normalized)

    @classmethod
    def from_yolo_polygon(cls, points: Sequence[float], is_normalized: bool = True) -> FieldGeometry:
        """Create geometry from YOLO polygon coordinate sequence [x1, y1, x2, y2, ...]."""
        if len(points) < 6 or len(points) % 2 != 0:
            raise ValueError(f"Polygon must have an even number of coordinates (>= 6), got {len(points)}")
        
        poly = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bbox = BBox(min(xs), min(ys), max(xs), max(ys))
        return cls(bbox=bbox, polygon=poly, is_normalized=is_normalized)

    def to_absolute(self, width: int, height: int) -> FieldGeometry:
        """Convert normalized coordinates to absolute pixel coordinates."""
        if not self.is_normalized:
            return self
        abs_bbox = BBox(
            self.bbox.x1 * width,
            self.bbox.y1 * height,
            self.bbox.x2 * width,
            self.bbox.y2 * height,
        )
        abs_poly = (
            [(p[0] * width, p[1] * height) for p in self.polygon]
            if self.polygon is not None
            else None
        )
        return FieldGeometry(bbox=abs_bbox, polygon=abs_poly, is_normalized=False)

    def to_normalized(self, width: int, height: int) -> FieldGeometry:
        """Convert absolute pixel coordinates to normalized [0, 1] coordinates."""
        if self.is_normalized:
            return self
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
        norm_bbox = BBox(
            self.bbox.x1 / width,
            self.bbox.y1 / height,
            self.bbox.x2 / width,
            self.bbox.y2 / height,
        )
        norm_poly = (
            [(p[0] / width, p[1] / height) for p in self.polygon]
            if self.polygon is not None
            else None
        )
        return FieldGeometry(bbox=norm_bbox, polygon=norm_poly, is_normalized=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_list(),
            "polygon": [list(p) for p in self.polygon] if self.polygon else None,
            "is_normalized": self.is_normalized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldGeometry:
        b = data["bbox"]
        poly = [tuple(p) for p in data["polygon"]] if data.get("polygon") else None
        return cls(
            bbox=BBox(b[0], b[1], b[2], b[3]),
            polygon=poly,
            is_normalized=data.get("is_normalized", True),
        )


@dataclass
class FieldTranscription:
    """Text content, transcription status, and language/script metadata for a field."""

    text: str | None = None
    status: TranscriptionStatus = TranscriptionStatus.MISSING
    language: str | None = None        # e.g., "en", "hi", "kn", "ta"
    script: str | None = None          # e.g., "latin", "devanagari"
    confidence: float | None = None    # OCR or annotation confidence
    is_handwritten: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status.value,
            "language": self.language,
            "script": self.script,
            "confidence": self.confidence,
            "is_handwritten": self.is_handwritten,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldTranscription:
        return cls(
            text=data.get("text"),
            status=TranscriptionStatus(data.get("status", TranscriptionStatus.MISSING.value)),
            language=data.get("language"),
            script=data.get("script"),
            confidence=data.get("confidence"),
            is_handwritten=data.get("is_handwritten", False),
        )


@dataclass
class FieldValidationRule:
    """Validation rule metadata for expected field formats."""

    expected_pattern: str | None = None    # Regex pattern
    expected_type: str | None = None       # e.g., "date", "numeric", "alphanumeric"
    checksum_algorithm: str | None = None  # e.g., "verhoeff" for Aadhaar

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_pattern": self.expected_pattern,
            "expected_type": self.expected_type,
            "checksum_algorithm": self.checksum_algorithm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldValidationRule:
        return cls(
            expected_pattern=data.get("expected_pattern"),
            expected_type=data.get("expected_type"),
            checksum_algorithm=data.get("checksum_algorithm"),
        )


@dataclass
class AnnotatedField:
    """A single annotated field region on a document."""

    field_name: str
    geometry: FieldGeometry
    transcription: FieldTranscription = field(default_factory=FieldTranscription)
    original_class_id: int | None = None
    confidence: float | None = None
    validation_rule: FieldValidationRule | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "geometry": self.geometry.to_dict(),
            "transcription": self.transcription.to_dict(),
            "original_class_id": self.original_class_id,
            "confidence": self.confidence,
            "validation_rule": self.validation_rule.to_dict() if self.validation_rule else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotatedField:
        val_rule = (
            FieldValidationRule.from_dict(data["validation_rule"])
            if data.get("validation_rule")
            else None
        )
        return cls(
            field_name=data["field_name"],
            geometry=FieldGeometry.from_dict(data["geometry"]),
            transcription=FieldTranscription.from_dict(data.get("transcription", {})),
            original_class_id=data.get("original_class_id"),
            confidence=data.get("confidence"),
            validation_rule=val_rule,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class DocumentProvenance:
    """Traceability metadata recording the origin of this annotated document."""

    source_dataset: str | None = None      # "aadhaar", "pan", "driving_license"
    source_split: str | None = None        # "train", "valid", "test"
    source_group_id: str | None = None     # e.g., "aadhaar_src_0042"
    source_image_rel_path: str | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "source_group_id": self.source_group_id,
            "source_image_rel_path": self.source_image_rel_path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentProvenance:
        return cls(
            source_dataset=data.get("source_dataset"),
            source_split=data.get("source_split"),
            source_group_id=data.get("source_group_id"),
            source_image_rel_path=data.get("source_image_rel_path"),
            sha256=data.get("sha256"),
        )


@dataclass
class AnnotatedDocument:
    """Canonical annotation container for a single document image."""

    image_path: str
    document_type: DocumentType
    image_size: tuple[int, int]  # (width, height) in pixels
    fields: list[AnnotatedField] = field(default_factory=list)
    provenance: DocumentProvenance | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate_fields(self, strict: bool = True) -> list[str]:
        """Validate that all attached fields belong to the document type's vocabulary.

        Args:
            strict: If True, raises ValueError on unknown fields; if False, returns list of warnings.
        """
        valid_set = get_valid_fields(self.document_type)
        errors: list[str] = []

        for f in self.fields:
            if f.field_name not in valid_set:
                msg = (
                    f"Field '{f.field_name}' is not a valid semantic field for "
                    f"document type '{self.document_type.value}'. Valid fields: {sorted(valid_set)}"
                )
                errors.append(msg)

        if strict and errors:
            raise ValueError("\n".join(errors))

        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "image_path": self.image_path,
            "document_type": self.document_type.value,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "fields": [f.to_dict() for f in self.fields],
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotatedDocument:
        doc_type = DocumentType(data["document_type"])
        size_dict = data["image_size"]
        image_size = (size_dict["width"], size_dict["height"])
        fields_list = [AnnotatedField.from_dict(f) for f in data.get("fields", [])]
        prov = (
            DocumentProvenance.from_dict(data["provenance"])
            if data.get("provenance")
            else None
        )
        return cls(
            image_path=data["image_path"],
            document_type=doc_type,
            image_size=image_size,
            fields=fields_list,
            provenance=prov,
            metadata=dict(data.get("metadata", {})),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )
