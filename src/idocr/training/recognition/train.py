"""Text recognition (OCR) training (placeholder).

TODO(owner): choose the recognition architecture and character set after
the audit establishes which scripts appear in transcriptions.
TODO: crop dataset built from detection labels + transcriptions, CER/WER
validation via idocr.evaluation.recognition.
"""

from __future__ import annotations

from typing import Any, Mapping


def train(config: Mapping[str, Any]) -> None:
    raise NotImplementedError("Text recognition (OCR) training is not implemented yet.")
