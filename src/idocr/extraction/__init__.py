"""Document-specific field extraction: OCR/layout results -> structured fields."""

from idocr.extraction.aadhaar import AadhaarExtractor
from idocr.extraction.base import FieldExtractor
from idocr.extraction.driving_license import DrivingLicenseExtractor
from idocr.extraction.pan import PanExtractor
from idocr.types import DocumentType


def default_extractors() -> dict[DocumentType, FieldExtractor]:
    """One extractor per supported document type."""
    return {
        DocumentType.AADHAAR: AadhaarExtractor(),
        DocumentType.PAN: PanExtractor(),
        DocumentType.DRIVING_LICENSE: DrivingLicenseExtractor(),
    }


__all__ = [
    "AadhaarExtractor",
    "DrivingLicenseExtractor",
    "FieldExtractor",
    "PanExtractor",
    "default_extractors",
]
