"""Document classifier interface: image -> document type.

TODO(owner): architecture choice (e.g. small CNN fine-tune vs. layout/keyword
heuristics on OCR output) after seeing dataset sizes and image variety.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from idocr.types import ClassificationResult, ImageArray


class DocumentClassifier(ABC):
    @abstractmethod
    def classify(self, image: ImageArray) -> ClassificationResult:
        """Predict the document type. Should return DocumentType.UNKNOWN when unsure."""
