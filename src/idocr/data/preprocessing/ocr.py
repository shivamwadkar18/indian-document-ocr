"""OCR crop preprocessing utilities.

The preprocessing baseline is intentionally conservative:
grayscale conversion, aspect-ratio-preserving height normalization,
and padding. More aggressive enhancement is deferred until OCR
training/evaluation provides evidence that it is beneficial.
"""

from __future__ import annotations

from PIL import Image, ImageOps


DEFAULT_TARGET_HEIGHT = 48
DEFAULT_HORIZONTAL_PADDING = 8
DEFAULT_VERTICAL_PADDING = 4


def preprocess_ocr_crop(
    crop: Image.Image,
    *,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    horizontal_padding: int = DEFAULT_HORIZONTAL_PADDING,
    vertical_padding: int = DEFAULT_VERTICAL_PADDING,
) -> Image.Image:
    """Prepare one detected text-field crop for OCR.

    The original aspect ratio is preserved.

    Returns:
        A grayscale PIL image whose resized text height is
        ``target_height`` plus vertical padding, with horizontal
        padding added on both sides.

    Raises:
        TypeError: if ``crop`` is not a PIL image.
        ValueError: for invalid preprocessing parameters or an
            invalid crop size.
    """
    if not isinstance(crop, Image.Image):
        raise TypeError("crop must be a PIL.Image.Image")

    if target_height <= 0:
        raise ValueError("target_height must be greater than zero")

    if horizontal_padding < 0:
        raise ValueError(
            "horizontal_padding must be non-negative"
        )

    if vertical_padding < 0:
        raise ValueError(
            "vertical_padding must be non-negative"
        )

    width, height = crop.size

    if width <= 0 or height <= 0:
        raise ValueError("crop must have positive width and height")

    # RGB/RGBA/etc. -> grayscale.
    gray = ImageOps.grayscale(crop)

    # Preserve aspect ratio while normalizing text height.
    new_width = max(
        1,
        round(width * target_height / height),
    )

    resized = gray.resize(
        (new_width, target_height),
        Image.Resampling.LANCZOS,
    )

    # Add breathing room around the text.
    padded = ImageOps.expand(
        resized,
        border=(
            horizontal_padding,
            vertical_padding,
            horizontal_padding,
            vertical_padding,
        ),
        fill=255,
    )

    return padded
