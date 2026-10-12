"""Unit tests for field validation (Aadhaar, PAN, Date of Birth, Verhoeff).

All tests use strictly fabricated, synthetic test data and run without
models, weights, external services, or real identity documents.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from idocr.extraction import AadhaarExtractor, PanExtractor
from idocr.models import TextDetector, TextRecognizer
from idocr.pipeline import DocumentPipeline
from idocr.types import (
    BBox,
    DocumentType,
    ExtractedField,
    RecognitionResult,
    TextRegion,
    ValidationIssue,
    ValidationResult,
)
from idocr.validation import (
    AadhaarValidator,
    PanValidator,
    compute_verhoeff_check_digit,
    default_validators,
    validate_aadhaar_number,
    validate_date_of_birth,
    validate_pan_number,
    validate_verhoeff,
)


# Helper to build a fabricated valid 12-digit Aadhaar number
def make_fabricated_aadhaar(base_11: str = "23456789012") -> str:
    chk = compute_verhoeff_check_digit(base_11)
    return f"{base_11}{chk}"


# ---------------------------------------------------------------------------
# 1. Verhoeff Checksum Algorithm Tests
# ---------------------------------------------------------------------------


def test_verhoeff_known_vectors():
    # Known test vectors for Verhoeff check digit calculation
    assert compute_verhoeff_check_digit("236") == 3
    assert compute_verhoeff_check_digit("142857") == 0
    assert compute_verhoeff_check_digit("12345") == 1

    assert validate_verhoeff("2363") is True
    assert validate_verhoeff("1428570") is True
    assert validate_verhoeff("123451") is True


def test_verhoeff_detects_single_digit_alteration():
    valid = make_fabricated_aadhaar("98765432109")
    assert validate_verhoeff(valid) is True

    # Mutate every single digit to a different digit
    for i in range(len(valid)):
        original_digit = int(valid[i])
        for delta in range(1, 10):
            altered_digit = (original_digit + delta) % 10
            mutated = valid[:i] + str(altered_digit) + valid[i + 1:]
            assert validate_verhoeff(mutated) is False, f"Failed to detect change at pos {i}"


def test_verhoeff_detects_adjacent_transpositions():
    valid = make_fabricated_aadhaar("54321098765")
    assert validate_verhoeff(valid) is True

    # Transpose every adjacent pair of distinct digits
    for i in range(len(valid) - 1):
        if valid[i] != valid[i + 1]:
            transposed = valid[:i] + valid[i + 1] + valid[i] + valid[i + 2:]
            assert validate_verhoeff(transposed) is False, f"Failed to detect transposition at {i},{i+1}"


def test_verhoeff_invalid_inputs():
    assert validate_verhoeff("") is False
    assert validate_verhoeff("abc") is False
    assert validate_verhoeff("1234 5678") is False  # contains space
    with pytest.raises(ValueError):
        compute_verhoeff_check_digit("")
    with pytest.raises(ValueError):
        compute_verhoeff_check_digit("12a45")


# ---------------------------------------------------------------------------
# 2. Aadhaar Number Validation Tests
# ---------------------------------------------------------------------------


def test_aadhaar_validation_valid_formats():
    num = make_fabricated_aadhaar("23456789012")
    spaced = f"{num[:4]} {num[4:8]} {num[8:]}"

    # Contiguous 12 digits
    assert validate_aadhaar_number(num) == []

    # 4-4-4 spacing
    assert validate_aadhaar_number(spaced) == []

    # Surrounding whitespace is handled safely
    assert validate_aadhaar_number(f"  {spaced}  ") == []
    assert validate_aadhaar_number(f"\t{num}\n") == []


def test_aadhaar_validation_checksum_mismatch():
    base = "23456789012"
    correct_chk = compute_verhoeff_check_digit(base)
    wrong_chk = (correct_chk + 1) % 10
    bad_aadhaar = f"{base[:4]} {base[4:8]} {base[8:]}{wrong_chk}"

    issues = validate_aadhaar_number(bad_aadhaar)
    assert len(issues) == 1
    assert issues[0].code == "aadhaar_checksum_mismatch"
    assert issues[0].field == "aadhaar_number"
    # Verify privacy: raw Aadhaar number must NOT appear in issue message
    assert bad_aadhaar not in issues[0].message
    assert base not in issues[0].message


def test_aadhaar_validation_malformed_formats():
    cases = [
        "12345678901",       # 11 digits
        "1234567890123",     # 13 digits
        "2345-6789-0123",    # hyphens instead of spaces
        "234 56789 0123",    # irregular spacing
        "2345 ABCD 0123",    # letters present
        "23456789012A",      # trailing letter
        "not_an_aadhaar",
    ]
    for bad in cases:
        issues = validate_aadhaar_number(bad)
        assert len(issues) == 1, f"Expected 1 issue for {bad}"
        assert issues[0].code == "invalid_aadhaar_format"
        assert issues[0].field == "aadhaar_number"
        assert bad not in issues[0].message


def test_aadhaar_validation_missing_and_empty():
    for empty in [None, "", "   ", "\t\n"]:
        issues = validate_aadhaar_number(empty)
        assert len(issues) == 1
        assert issues[0].code == "missing_field"
        assert issues[0].field == "aadhaar_number"


# ---------------------------------------------------------------------------
# 3. PAN Number Validation Tests
# ---------------------------------------------------------------------------


def test_pan_validation_valid():
    # Individual PAN format: 3 letters + 'P' + 1 letter + 4 digits + 1 letter
    valid_samples = [
        "ABCPE1234F",
        "xyzpe1234k",            # lowercase variation
        "AbCpE9876Z",            # mixed case variation
        "  ABCPE1234F  ",        # surrounding whitespace
    ]
    for s in valid_samples:
        assert validate_pan_number(s) == [], f"Expected valid for {s}"


def test_pan_validation_invalid_holder_type():
    # In individual PAN, 4th char must be 'P' (Person/Individual)
    invalid_holders = [
        "ABCCE1234F",  # 'C' = Company
        "ABCHE1234F",  # 'H' = HUF
        "ABCFE1234F",  # 'F' = Firm
        "ABCAE1234F",  # 'A' = AOP
        "ABC1E1234F",  # digit in holder position
    ]
    for bad in invalid_holders:
        issues = validate_pan_number(bad)
        assert len(issues) == 1
        assert issues[0].code == "invalid_pan_format"
        assert issues[0].field == "pan_number"
        # Privacy check: raw PAN must not be in issue message
        assert bad not in issues[0].message


def test_pan_validation_malformed_formats():
    cases = [
        "ABCPE123F",     # 9 chars (too short)
        "ABCPE12345F",   # 11 chars (too long)
        "12CPE1234F",    # digits in prefix
        "ABCPE12A4F",    # letter in numeric section
        "ABCPE12345",    # digit at the end
        "ABCPE 1234F",   # internal space
        "not_a_pan",
    ]
    for bad in cases:
        issues = validate_pan_number(bad)
        assert len(issues) == 1, f"Expected 1 issue for {bad}"
        assert issues[0].code == "invalid_pan_format"
        assert issues[0].field == "pan_number"
        assert bad not in issues[0].message


def test_pan_validation_missing_and_empty():
    for empty in [None, "", "   ", "\t\n"]:
        issues = validate_pan_number(empty)
        assert len(issues) == 1
        assert issues[0].code == "missing_field"
        assert issues[0].field == "pan_number"


# ---------------------------------------------------------------------------
# 4. Date of Birth Validation Tests
# ---------------------------------------------------------------------------


def test_dob_validation_valid():
    assert validate_date_of_birth("15/08/1990") == []
    assert validate_date_of_birth("29/02/2024") == []  # Leap year
    assert validate_date_of_birth("  01/01/2000  ") == []


def test_dob_validation_impossible_calendar_dates():
    impossible = [
        "31/02/1990",  # Feb 31 does not exist
        "29/02/2023",  # 2023 is not a leap year
        "31/04/2000",  # April has 30 days
        "31/06/1985",  # June has 30 days
        "31/11/1995",  # November has 30 days
        "00/01/2000",  # Day 0
        "15/13/2000",  # Month 13
        "15/00/2000",  # Month 0
        "32/01/2000",  # Day 32
    ]
    for bad in impossible:
        issues = validate_date_of_birth(bad)
        assert len(issues) == 1, f"Expected impossible_date for {bad}"
        assert issues[0].code == "impossible_date"
        assert issues[0].field == "date_of_birth"


def test_dob_validation_malformed_formats():
    malformed = [
        "1990-08-15",   # ISO format not in OCR vocabulary
        "15-08-1990",   # Hyphen format
        "1/1/2000",     # Missing leading zeros
        "15/08/90",     # 2-digit year
        "15/08/19900",  # 5-digit year
        "invalid_date",
    ]
    for bad in malformed:
        issues = validate_date_of_birth(bad)
        assert len(issues) == 1, f"Expected invalid_date_format for {bad}"
        assert issues[0].code == "invalid_date_format"
        assert issues[0].field == "date_of_birth"


def test_dob_validation_future_date():
    ref = date(2025, 1, 1)
    # Past relative to ref
    assert validate_date_of_birth("31/12/2024", reference_date=ref) == []
    # Same day as ref
    assert validate_date_of_birth("01/01/2025", reference_date=ref) == []
    # Future relative to ref
    future_issues = validate_date_of_birth("02/01/2025", reference_date=ref)
    assert len(future_issues) == 1
    assert future_issues[0].code == "future_date_of_birth"
    assert future_issues[0].field == "date_of_birth"

    # Distant future with default date.today()
    assert len(validate_date_of_birth("01/01/2099")) == 1
    assert validate_date_of_birth("01/01/2099")[0].code == "future_date_of_birth"


def test_dob_validation_missing_and_empty():
    for empty in [None, "", "   "]:
        issues = validate_date_of_birth(empty)
        assert len(issues) == 1
        assert issues[0].code == "missing_field"
        assert issues[0].field == "date_of_birth"


# ---------------------------------------------------------------------------
# 5. AadhaarValidator and PanValidator Document-Level Tests
# ---------------------------------------------------------------------------


def test_aadhaar_validator_success():
    validator = AadhaarValidator()
    num = make_fabricated_aadhaar("23456789012")
    fields = {
        "aadhaar_number": ExtractedField("aadhaar_number", f"{num[:4]} {num[4:8]} {num[8:]}"),
        "date_of_birth": ExtractedField("date_of_birth", "10/05/1988"),
        "name": ExtractedField("name", "ANIL SHARMA"),
    }
    result = validator.validate(fields)
    assert result.is_valid is True
    assert result.issues == []


def test_aadhaar_validator_multiple_issues():
    validator = AadhaarValidator()
    fields = {
        "aadhaar_number": ExtractedField("aadhaar_number", "1234 5678 9012"),  # Bad checksum
        "date_of_birth": ExtractedField("date_of_birth", "31/02/1990"),        # Impossible date
    }
    result = validator.validate(fields)
    assert result.is_valid is False
    assert len(result.issues) == 2
    codes = {i.code for i in result.issues}
    assert codes == {"aadhaar_checksum_mismatch", "impossible_date"}


def test_pan_validator_success():
    validator = PanValidator()
    fields = {
        "pan_number": ExtractedField("pan_number", "ABCPE1234F"),
        "date_of_birth": ExtractedField("date_of_birth", "25/12/1992"),
    }
    result = validator.validate(fields)
    assert result.is_valid is True
    assert result.issues == []


def test_pan_validator_multiple_issues():
    validator = PanValidator()
    fields = {
        "pan_number": ExtractedField("pan_number", "ABCCE1234F"),       # Non-individual ('C')
        "date_of_birth": ExtractedField("date_of_birth", "01/01/2099"),  # Future date
    }
    result = validator.validate(fields)
    assert result.is_valid is False
    assert len(result.issues) == 2
    codes = {i.code for i in result.issues}
    assert codes == {"invalid_pan_format", "future_date_of_birth"}


def test_validators_handle_null_and_missing_fields():
    av = AadhaarValidator()
    # Missing aadhaar_number field key
    res1 = av.validate({})
    assert res1.is_valid is False
    assert any(i.code == "missing_field" and i.field == "aadhaar_number" for i in res1.issues)

    # ExtractedField with value=None
    res2 = av.validate({
        "aadhaar_number": ExtractedField("aadhaar_number", None),
        "date_of_birth": ExtractedField("date_of_birth", None),
    })
    assert res2.is_valid is False
    assert len(res2.issues) == 2
    assert all(i.code == "missing_field" for i in res2.issues)

    pv = PanValidator()
    res3 = pv.validate({"pan_number": ExtractedField("pan_number", None)})
    assert res3.is_valid is False
    assert res3.issues[0].code == "missing_field"


def test_validation_does_not_mutate_field_values():
    pv = PanValidator()
    raw_pan = "  abcpe1234f  "
    fields = {"pan_number": ExtractedField("pan_number", raw_pan)}
    result = pv.validate(fields)
    assert result.is_valid is True
    # Crucial requirement: raw OCR text in ExtractedField must not be modified
    assert fields["pan_number"].value == raw_pan


# ---------------------------------------------------------------------------
# 6. Pipeline Integration & Fault Tolerance Tests
# ---------------------------------------------------------------------------


class FixedDetector(TextDetector):
    def __init__(self, regions: list[TextRegion]):
        self.regions = regions

    def detect(self, image):
        return list(self.regions)


class FixedRecognizer(TextRecognizer):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def recognize(self, crop):
        raise AssertionError("recognize_batch must be used")

    def recognize_batch(self, crops):
        return [RecognitionResult(t, 0.95) for t in self.texts[:len(crops)]]


from idocr.data.annotation_schema import FIELD_NAME_TO_DETECTOR_CLASS


def _make_region(name: str, box: tuple[float, float, float, float], conf: float = 0.9) -> TextRegion:
    return TextRegion(
        BBox(*box),
        conf,
        class_id=FIELD_NAME_TO_DETECTOR_CLASS[name],
        class_name=name,
    )


def test_pipeline_with_validators_continuing_normally_on_invalid_field():
    # Build a simulated PAN card with invalid PAN format
    regions = [
        _make_region("name", (0, 0, 100, 20)),
        _make_region("fathers_name", (0, 20, 100, 40)),
        _make_region("date_of_birth", (0, 40, 100, 60)),
        _make_region("pan_number", (0, 60, 100, 80)),
    ]
    texts = [
        "ROHAN SHARMA",
        "SURESH SHARMA",
        "15/08/1990",       # Valid DOB
        "BAD_PAN_FORMAT",   # Invalid PAN
    ]

    pipe = DocumentPipeline(
        detector=FixedDetector(regions),
        recognizer=FixedRecognizer(texts),
        extractors={DocumentType.PAN: PanExtractor()},
        validators=default_validators(),
    )

    dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = pipe.run(dummy_image, document_type=DocumentType.PAN)

    # Pipeline completed normally without exceptions
    assert result.document_type is DocumentType.PAN
    assert result.fields["pan_number"].value == "BAD_PAN_FORMAT"
    assert result.fields["date_of_birth"].value == "15/08/1990"

    # Validation captured the issue
    assert result.validation is not None
    assert result.validation.is_valid is False
    assert len(result.validation.issues) == 1
    assert result.validation.issues[0].code == "invalid_pan_format"
    assert result.validation.issues[0].field == "pan_number"

    # JSON serialization is clean
    d = result.to_dict()
    assert d["validation"]["is_valid"] is False
    assert d["validation"]["issues"][0]["code"] == "invalid_pan_format"
    json.dumps(d)


def test_pipeline_with_validators_passes_on_valid_document():
    num = make_fabricated_aadhaar("33334444555")
    spaced_num = f"{num[:4]} {num[4:8]} {num[8:]}"

    regions = [
        _make_region("name", (0, 0, 100, 20)),
        _make_region("date_of_birth", (0, 20, 100, 40)),
        _make_region("gender", (0, 40, 100, 60)),
        _make_region("aadhaar_number", (0, 60, 100, 80)),
    ]
    texts = [
        "PRIYA PATEL",
        "05/11/1995",
        "FEMALE",
        spaced_num,
    ]

    pipe = DocumentPipeline(
        detector=FixedDetector(regions),
        recognizer=FixedRecognizer(texts),
        extractors={DocumentType.AADHAAR: AadhaarExtractor()},
        validators=default_validators(),
    )

    dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = pipe.run(dummy_image, document_type=DocumentType.AADHAAR)

    assert result.validation is not None
    assert result.validation.is_valid is True
    assert result.validation.issues == []
