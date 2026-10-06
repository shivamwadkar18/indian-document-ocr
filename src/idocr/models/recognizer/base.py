"""Text recognizer interface: text crop -> text + confidence.

TODO(owner): architecture and character set depend on the audited
transcriptions (English only vs. Hindi/regional scripts on Aadhaar).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from idocr.types import ImageArray, RecognitionResult


class TextRecognizer(ABC):
    @abstractmethod
    def recognize(self, crop: ImageArray) -> RecognitionResult:
        """Recognize the text in a single cropped region."""

    def recognize_batch(self, crops: Sequence[ImageArray]) -> list[RecognitionResult]:
        """Default: one at a time. Override for batched inference."""
        return [self.recognize(c) for c in crops]
