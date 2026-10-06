"""Aadhaar field extraction (not implemented).

TODO(after audit):
  - Field set: confirm against the dataset's labels (which fields are annotated,
    front/back sides, whether field-level classes exist).
  - Strategy: rules over OCR text + layout, vs. learned field classification,
    depending on whether the detector is trained with field classes.
  - Privacy: decide whether/how the Aadhaar number is masked in outputs.
"""

from __future__ import annotations

from typing import Sequence

from idocr.extraction.base import FieldExtractor
from idocr.types import DocumentType, ExtractedField, RecognizedRegion


class AadhaarExtractor(FieldExtractor):
    document_type = DocumentType.AADHAAR

    def extract(
        self, regions: Sequence[RecognizedRegion], image_size: tuple[int, int]
    ) -> dict[str, ExtractedField]:
        raise NotImplementedError("Aadhaar extraction rules are defined after the dataset audit.")
