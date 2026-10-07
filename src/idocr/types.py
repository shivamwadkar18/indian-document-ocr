"""Core types shared across the inference pipeline.

These describe *pipeline outputs* (what components produce at inference time).
They are deliberately separate from the training *annotation schema* in
``idocr.data.annotation_schema``, which stays a draft until the source datasets
have been audited.

Every result type has ``to_dict()`` returning JSON-serialisable data, so a
future backend can expose ``image -> PipelineResult -> JSON`` without importing
any ML framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Image convention used by every component interface:
#: H x W x 3, dtype uint8, RGB channel order.
ImageArray = NDArray[np.uint8]


class DocumentType(str, Enum):
    AADHAAR = "aadhaar"
    PAN = "pan"
    DRIVING_LICENSE = "driving_license"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in absolute pixel coordinates.

    (x1, y1) is the top-left corner, (x2, y2) the bottom-right corner.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid bbox (x2 < x1 or y2 < y1): {self.to_list()}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class TextRegion:
    """Output of the field detector for one detected region."""

    bbox: BBox
    confidence: float
    # Detector class identity. ``class_name`` is the canonical field name
    # such as ``name`` or ``pan_number``.
    class_id: int = -1
    class_name: str | None = None
    # Reserved for future polygon-capable detectors.
    polygon: list[tuple[float, float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "polygon": [list(p) for p in self.polygon] if self.polygon else None,
        }


@dataclass
class RecognitionResult:
    """Output of the text recognizer for one crop."""

    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "confidence": self.confidence}


@dataclass
class RecognizedRegion:
    """A detected region paired with its recognized text."""

    region: TextRegion
    recognition: RecognitionResult

    def to_dict(self) -> dict[str, Any]:
        return {**self.region.to_dict(), **self.recognition.to_dict()}


@dataclass
class ClassificationResult:
    document_type: DocumentType
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type.value,
            "confidence": self.confidence,
            "scores": dict(self.scores),
        }


@dataclass
class ExtractedField:
    """One structured field extracted from a document."""

    name: str
    value: str | None
    confidence: float | None = None
    # Indices into PipelineResult.regions that contributed to this field.
    source_regions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "source_regions": list(self.source_regions),
        }


@dataclass
class ValidationIssue:
    code: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "field": self.field}


@dataclass
class ValidationResult:
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"is_valid": self.is_valid, "issues": [i.to_dict() for i in self.issues]}


@dataclass
class PipelineResult:
    """Final structured output of ``DocumentPipeline.run``."""

    document_type: DocumentType
    image_size: tuple[int, int]  # (width, height) of the image the regions refer to
    classification: ClassificationResult | None = None
    regions: list[RecognizedRegion] = field(default_factory=list)
    fields: dict[str, ExtractedField] = field(default_factory=dict)
    validation: ValidationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type.value,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "classification": self.classification.to_dict() if self.classification else None,
            "regions": [r.to_dict() for r in self.regions],
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "validation": self.validation.to_dict() if self.validation else None,
        }
