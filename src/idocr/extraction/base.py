"""Field extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

from idocr.types import DocumentType, ExtractedField, RecognizedRegion


class FieldExtractor(ABC):
    document_type: ClassVar[DocumentType]

    @abstractmethod
    def extract(
        self, regions: Sequence[RecognizedRegion], image_size: tuple[int, int]
    ) -> dict[str, ExtractedField]:
        """Map recognized regions (pixel coords of an image of ``image_size`` = (w, h))
        to named fields."""
