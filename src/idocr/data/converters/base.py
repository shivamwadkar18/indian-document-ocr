"""Converter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from idocr.data.annotation_schema import AnnotatedDocument


class DatasetConverter(ABC):
    """Reads a raw dataset (read-only) and yields unified annotations.

    Converters must never write into the source directory; persisting the
    converted output (under data/processed/) is the caller's job.
    """

    #: Identifier of the source format handled, e.g. "aadhaar_v1".
    source_format: str = "base"

    @abstractmethod
    def convert(self, source_dir: Path) -> Iterator[AnnotatedDocument]:
        """Yield one ``AnnotatedDocument`` per source image."""
