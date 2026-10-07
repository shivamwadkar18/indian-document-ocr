"""Text recognizer interface and implementations.

``idocr.models.recognizer.crnn`` (CRNN + CTC) requires PyTorch and is not
imported here so the package stays importable without it.
"""

from idocr.models.recognizer.base import TextRecognizer
from idocr.models.recognizer.vocab import BLANK_INDEX, DEFAULT_ALPHABET, Vocabulary

__all__ = ["BLANK_INDEX", "DEFAULT_ALPHABET", "TextRecognizer", "Vocabulary"]
