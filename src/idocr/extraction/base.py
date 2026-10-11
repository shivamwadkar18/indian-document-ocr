"""Field extractor interface and the shared detector-class field selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

from idocr.data.annotation_schema import DETECTOR_CLASSES, get_valid_fields
from idocr.types import DocumentType, ExtractedField, RecognizedRegion


class FieldExtractor(ABC):
    document_type: ClassVar[DocumentType]

    @abstractmethod
    def extract(
        self, regions: Sequence[RecognizedRegion], image_size: tuple[int, int]
    ) -> dict[str, ExtractedField]:
        """Map recognized regions (pixel coords of an image of ``image_size`` = (w, h))
        to named fields."""


def select_detected_fields(
    regions: Sequence[RecognizedRegion], field_names: set[str]
) -> dict[str, ExtractedField]:
    """One field per name from a field-level detector's class labels.

    For each field the detection with the highest detection confidence and
    non-empty recognized text is used; ``confidence`` is detection x
    recognition confidence. A field with no usable detection is returned with
    ``value=None`` (never guessed). Fields keep the detector class order.
    """
    fields: dict[str, ExtractedField] = {}
    for name in (n for n in DETECTOR_CLASSES.values() if n in field_names):
        candidates = [
            i for i, r in enumerate(regions)
            if r.region.class_name == name and r.recognition.text
        ]
        if not candidates:
            fields[name] = ExtractedField(name=name, value=None)
            continue
        best = max(candidates, key=lambda i: (regions[i].region.confidence, regions[i].recognition.confidence))
        region = regions[best]
        fields[name] = ExtractedField(
            name=name,
            value=region.recognition.text,
            confidence=region.region.confidence * region.recognition.confidence,
            source_regions=[best],
        )
    return fields


class DetectedFieldExtractor(FieldExtractor):
    """Extractor for documents whose fields come straight from detector classes."""

    def extract(
        self, regions: Sequence[RecognizedRegion], image_size: tuple[int, int]
    ) -> dict[str, ExtractedField]:
        return select_detected_fields(regions, get_valid_fields(self.document_type))
