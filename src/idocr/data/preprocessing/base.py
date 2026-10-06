"""Preprocessing interface.

TODO(after audit): concrete steps (resize policy, deskew, orientation fix,
document cropping, denoising) depend on what the real images look like —
scans vs phone photos, resolution range, background clutter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from idocr.types import ImageArray


class Preprocessor(ABC):
    """image -> image. Geometric steps change coordinates; downstream regions
    refer to the *preprocessed* image."""

    @abstractmethod
    def __call__(self, image: ImageArray) -> ImageArray: ...


class IdentityPreprocessor(Preprocessor):
    def __call__(self, image: ImageArray) -> ImageArray:
        return image


class Compose(Preprocessor):
    def __init__(self, steps: Sequence[Preprocessor]) -> None:
        self.steps = list(steps)

    def __call__(self, image: ImageArray) -> ImageArray:
        for step in self.steps:
            image = step(image)
        return image
