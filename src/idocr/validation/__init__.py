"""Validation of extracted fields.

Provides field-level validators and algorithms (Verhoeff checksum, PAN structure,
and calendar date validation) for Indian identity documents (Aadhaar, PAN).
"""

from __future__ import annotations

from idocr.types import DocumentType
from idocr.validation.aadhaar import AadhaarValidator
from idocr.validation.base import Validator
from idocr.validation.pan import PanValidator
from idocr.validation.rules import (
    validate_aadhaar_number,
    validate_date_of_birth,
    validate_pan_number,
)
from idocr.validation.verhoeff import (
    compute_verhoeff_check_digit,
    validate_verhoeff,
)


def default_validators() -> dict[DocumentType, Validator]:
    """One validator per supported document type."""
    return {
        DocumentType.AADHAAR: AadhaarValidator(),
        DocumentType.PAN: PanValidator(),
    }


__all__ = [
    "AadhaarValidator",
    "PanValidator",
    "Validator",
    "compute_verhoeff_check_digit",
    "default_validators",
    "validate_aadhaar_number",
    "validate_date_of_birth",
    "validate_pan_number",
    "validate_verhoeff",
]
