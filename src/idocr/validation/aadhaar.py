"""Aadhaar field validator.

Validates extracted fields for Aadhaar documents (Aadhaar number and DOB).
"""

from __future__ import annotations

from datetime import date
from typing import Mapping

from idocr.types import ExtractedField, ValidationIssue, ValidationResult
from idocr.validation.base import Validator
from idocr.validation.rules import validate_aadhaar_number, validate_date_of_birth


class AadhaarValidator(Validator):
    """Validator for Aadhaar documents."""

    def __init__(self, *, reference_date: date | None = None) -> None:
        self.reference_date = reference_date

    def validate(self, fields: Mapping[str, ExtractedField]) -> ValidationResult:
        issues: list[ValidationIssue] = []

        if "aadhaar_number" in fields:
            issues.extend(validate_aadhaar_number(fields["aadhaar_number"].value))
        else:
            issues.append(
                ValidationIssue(
                    code="missing_field",
                    message="aadhaar_number field is missing",
                    field="aadhaar_number",
                )
            )

        if "date_of_birth" in fields:
            dob_val = fields["date_of_birth"].value
            if dob_val is not None:
                issues.extend(
                    validate_date_of_birth(dob_val, reference_date=self.reference_date)
                )
            else:
                issues.append(
                    ValidationIssue(
                        code="missing_field",
                        message="date_of_birth is missing or empty",
                        field="date_of_birth",
                    )
                )

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)
