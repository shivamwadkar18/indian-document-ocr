"""Recognition-dataset sample schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecognitionSample:
    """One OCR training/evaluation sample."""

    image_path: str
    text: str
    field_name: str
    document_type: str
    source: str  # "synthetic" or "real"
    split: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text must not be empty")

        if self.source not in {"synthetic", "real"}:
            raise ValueError(
                "source must be 'synthetic' or 'real'"
            )

        if self.split not in {"train", "valid", "test"}:
            raise ValueError(
                "split must be 'train', 'valid', or 'test'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "text": self.text,
            "field_name": self.field_name,
            "document_type": self.document_type,
            "source": self.source,
            "split": self.split,
        }
