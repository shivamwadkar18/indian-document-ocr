"""Model interfaces. Concrete architectures are chosen after the dataset audit.

Interfaces are framework-agnostic (numpy in, dataclasses out) so the pipeline,
tests and a future backend do not depend on PyTorch.
"""

from idocr.models.classifier import DocumentClassifier
from idocr.models.detector import TextDetector
from idocr.models.recognizer import TextRecognizer

__all__ = ["DocumentClassifier", "TextDetector", "TextRecognizer"]
