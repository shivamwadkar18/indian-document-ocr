"""Augmentation interface.

TODO(after schema): geometric augmentations (rotation, perspective, crop) must
transform annotations together with the image. The signature will be extended
to take/return annotations once ``annotation_schema`` is finalised. Until then
only the photometric image-only interface is defined.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from idocr.types import ImageArray


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, image: ImageArray, rng: np.random.Generator) -> ImageArray: ...


class Compose(Augmentation):
    def __init__(self, steps: Sequence[Augmentation]) -> None:
        self.steps = list(steps)

    def __call__(self, image: ImageArray, rng: np.random.Generator) -> ImageArray:
        for step in self.steps:
            image = step(image, rng)
        return image
