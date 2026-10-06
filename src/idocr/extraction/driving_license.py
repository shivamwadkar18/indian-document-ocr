"""Driving Licence field extraction (not implemented).

TODO:
  - Field set is defined jointly with the synthetic DL generator, since
    training data for DL will be synthetic.
  - Handle multiple state layouts.
"""

from __future__ import annotations

from typing import Sequence

from idocr.extraction.base import FieldExtractor
from idocr.types import DocumentType, ExtractedField, RecognizedRegion


class DrivingLicenseExtractor(FieldExtractor):
    document_type = DocumentType.DRIVING_LICENSE

    def extract(
        self, regions: Sequence[RecognizedRegion], image_size: tuple[int, int]
    ) -> dict[str, ExtractedField]:
        raise NotImplementedError("Driving Licence extraction rules are not defined yet.")
