"""Text detector interface: image -> text regions.

TODO(owner): architecture choice depends on the audited labels — box vs polygon,
word vs line vs field granularity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from idocr.types import ImageArray, TextRegion


class TextDetector(ABC):
    @abstractmethod
    def detect(self, image: ImageArray) -> list[TextRegion]:
        """Return text regions in pixel coordinates of ``image``."""
