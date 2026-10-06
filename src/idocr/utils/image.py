"""Image loading and basic geometric helpers.

All images are handled as numpy arrays following ``idocr.types.ImageArray``
(H x W x 3, uint8, RGB).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from idocr.types import BBox, ImageArray


def load_image(path: str | Path) -> ImageArray:
    """Load an image file as an RGB uint8 array, applying EXIF orientation."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


def image_size(image: ImageArray) -> tuple[int, int]:
    """Return (width, height)."""
    return int(image.shape[1]), int(image.shape[0])


def crop(image: ImageArray, bbox: BBox) -> ImageArray:
    """Crop ``bbox`` from ``image``, clipped to image bounds."""
    w, h = image_size(image)
    x1 = max(0, min(w, int(np.floor(bbox.x1))))
    y1 = max(0, min(h, int(np.floor(bbox.y1))))
    x2 = max(0, min(w, int(np.ceil(bbox.x2))))
    y2 = max(0, min(h, int(np.ceil(bbox.y2))))
    return image[y1:y2, x1:x2]
