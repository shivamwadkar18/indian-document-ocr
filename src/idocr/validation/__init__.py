"""Validation of extracted fields.

TODO(owner): decide the rule set per document type. Candidate checks that do
not depend on the datasets: PAN format pattern, Aadhaar checksum (Verhoeff),
date sanity (DOB in the past, DL validity after issue date).
"""

from idocr.validation.base import Validator

__all__ = ["Validator"]
