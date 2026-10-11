"""Aadhaar field extraction from the field-level detector's classes.

Fields: ``name``, ``date_of_birth``, ``gender``, ``aadhaar_number``
(``DOCUMENT_FIELDS`` in ``idocr.data.annotation_schema``).

TODO(owner): decide whether/how the Aadhaar number is masked in outputs.
"""

from __future__ import annotations

from idocr.extraction.base import DetectedFieldExtractor
from idocr.types import DocumentType


class AadhaarExtractor(DetectedFieldExtractor):
    document_type = DocumentType.AADHAAR
