"""Field-level validation rules for Indian identity documents.

Contains individual validation routines for Aadhaar number, PAN number, and
Date of Birth.

NOTE: Validation routines only verify structural syntax and mathematical
consistency. They do not query government databases (UIDAI, Income Tax
Department) or guarantee document authenticity or identity registration.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from idocr.types import ValidationIssue
from idocr.validation.verhoeff import validate_verhoeff

# Aadhaar: exactly 12 digits, either contiguous or in 3 blocks of 4 digits separated by a space
AADHAAR_PATTERN = re.compile(r"^(\d{12}|\d{4} \d{4} \d{4})$")

# Individual PAN: 3 letters, 'P' (individual holder), 1 letter (surname initial), 4 digits, 1 letter
PAN_INDIVIDUAL_PATTERN = re.compile(r"^[A-Z]{3}P[A-Z]\d{4}[A-Z]$")

# Date of birth: DD/MM/YYYY format as supported by the OCR pipeline
DOB_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4}$")


def validate_aadhaar_number(value: Any) -> list[ValidationIssue]:
    """Validate a 12-digit Aadhaar number against format and Verhoeff checksum.

    Allows 12 contiguous digits or 3 space-separated 4-digit groups (e.g. 'XXXX XXXX XXXX').
    Surrounding whitespace is trimmed for validation without modifying the original text.

    Args:
        value: Extracted text to validate.

    Returns:
        List of ValidationIssue objects (empty if valid).
    """
    if value is None or not isinstance(value, str) or not value.strip():
        return [
            ValidationIssue(
                code="missing_field",
                message="Aadhaar number is missing or empty",
                field="aadhaar_number",
            )
        ]

    val = value.strip()
    if not AADHAAR_PATTERN.match(val):
        return [
            ValidationIssue(
                code="invalid_aadhaar_format",
                message="Aadhaar number must be 12 digits, optionally grouped in 4-digit blocks separated by spaces",
                field="aadhaar_number",
            )
        ]

    digits = val.replace(" ", "")
    if not validate_verhoeff(digits):
        return [
            ValidationIssue(
                code="aadhaar_checksum_mismatch",
                message="Aadhaar number failed Verhoeff checksum verification",
                field="aadhaar_number",
            )
        ]

    return []


def validate_pan_number(value: Any) -> list[ValidationIssue]:
    """Validate a 10-character individual PAN against expected syntax.

    Structure: 3 letters, 'P' (individual holder), 1 surname-initial letter,
    4 digits, and 1 final letter (e.g. 'ABCPE1234F').
    Case and surrounding whitespace are normalized during validation without
    modifying the original extracted text.

    Args:
        value: Extracted text to validate.

    Returns:
        List of ValidationIssue objects (empty if valid).
    """
    if value is None or not isinstance(value, str) or not value.strip():
        return [
            ValidationIssue(
                code="missing_field",
                message="PAN number is missing or empty",
                field="pan_number",
            )
        ]

    val = value.strip().upper()
    if not PAN_INDIVIDUAL_PATTERN.match(val):
        return [
            ValidationIssue(
                code="invalid_pan_format",
                message="PAN number must match individual PAN format (e.g. ABCPE1234F)",
                field="pan_number",
            )
        ]

    return []


def validate_date_of_birth(
    value: Any, *, reference_date: date | None = None
) -> list[ValidationIssue]:
    """Validate date of birth against supported DD/MM/YYYY format and calendar sanity.

    Rejects impossible calendar dates (e.g. 31/02/1990) and future dates.

    Args:
        value: Extracted text to validate.
        reference_date: Date to evaluate 'future' against (defaults to date.today()).

    Returns:
        List of ValidationIssue objects (empty if valid).
    """
    if value is None or not isinstance(value, str) or not value.strip():
        return [
            ValidationIssue(
                code="missing_field",
                message="Date of birth is missing or empty",
                field="date_of_birth",
            )
        ]

    val = value.strip()
    if not DOB_PATTERN.match(val):
        return [
            ValidationIssue(
                code="invalid_date_format",
                message="Date of birth must be in DD/MM/YYYY format",
                field="date_of_birth",
            )
        ]

    try:
        parsed_date = datetime.strptime(val, "%d/%m/%Y").date()
    except ValueError:
        return [
            ValidationIssue(
                code="impossible_date",
                message="Date of birth is not a valid calendar date",
                field="date_of_birth",
            )
        ]

    today = reference_date or date.today()
    if parsed_date > today:
        return [
            ValidationIssue(
                code="future_date_of_birth",
                message="Date of birth cannot be in the future",
                field="date_of_birth",
            )
        ]

    return []
