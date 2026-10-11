"""PAN field extraction from the field-level detector's classes.

Fields: ``name``, ``fathers_name``, ``date_of_birth``, ``pan_number``
(``DOCUMENT_FIELDS`` in ``idocr.data.annotation_schema``).
"""

from __future__ import annotations

from idocr.extraction.base import DetectedFieldExtractor
from idocr.types import DocumentType


class PanExtractor(DetectedFieldExtractor):
    document_type = DocumentType.PAN
