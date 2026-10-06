"""Validator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

from idocr.types import ExtractedField, ValidationResult


class Validator(ABC):
    @abstractmethod
    def validate(self, fields: Mapping[str, ExtractedField]) -> ValidationResult: ...
