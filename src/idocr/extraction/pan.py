"""PAN field extraction (not implemented).

TODO(after audit):
  - Field set: confirm against the dataset's labels.
  - Card layout variants present in the data (older vs. newer PAN designs).
"""

from __future__ import annotations

from typing import Sequence

from idocr.extraction.base import FieldExtractor
from idocr.types import DocumentType, ExtractedField, RecognizedRegion


class PanExtractor(FieldExtractor):
    document_type = DocumentType.PAN

    def extract(
        self, regions: Sequence[RecognizedRegion], image_size: tuple[int, int]
    ) -> dict[str, ExtractedField]:
        raise NotImplementedError("PAN extraction rules are defined after the dataset audit.")
