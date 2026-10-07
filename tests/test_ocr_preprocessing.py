import pytest
from PIL import Image

from idocr.data.preprocessing.ocr import (
    DEFAULT_HORIZONTAL_PADDING,
    DEFAULT_TARGET_HEIGHT,
    DEFAULT_VERTICAL_PADDING,
    preprocess_ocr_crop,
)


def test_preprocess_preserves_aspect_ratio() -> None:
    crop = Image.new("RGB", (200, 20), "black")

    result = preprocess_ocr_crop(crop)

    expected_inner_width = round(
        200 * DEFAULT_TARGET_HEIGHT / 20
    )

    expected_width = (
        expected_inner_width
        + 2 * DEFAULT_HORIZONTAL_PADDING
    )

    expected_height = (
        DEFAULT_TARGET_HEIGHT
        + 2 * DEFAULT_VERTICAL_PADDING
    )

    assert result.mode == "L"
    assert result.size == (
        expected_width,
        expected_height,
    )


def test_preprocess_adds_vertical_padding() -> None:
    crop = Image.new("RGB", (100, 20), "black")

    result = preprocess_ocr_crop(crop)

    assert result.height == (
        DEFAULT_TARGET_HEIGHT
        + 2 * DEFAULT_VERTICAL_PADDING
    )

    assert result.getpixel((0, 0)) == 255
    assert result.getpixel((0, result.height - 1)) == 255


def test_preprocess_adds_horizontal_padding() -> None:
    crop = Image.new("RGB", (100, 20), "black")

    result = preprocess_ocr_crop(crop)

    assert result.getpixel((0, 10)) == 255
    assert result.getpixel((result.width - 1, 10)) == 255


def test_preprocess_converts_to_grayscale() -> None:
    crop = Image.new("RGB", (100, 20), (20, 80, 200))

    result = preprocess_ocr_crop(crop)

    assert result.mode == "L"


def test_preprocess_rejects_invalid_parameters() -> None:
    crop = Image.new("RGB", (100, 20), "white")

    with pytest.raises(ValueError):
        preprocess_ocr_crop(
            crop,
            target_height=0,
        )

    with pytest.raises(ValueError):
        preprocess_ocr_crop(
            crop,
            horizontal_padding=-1,
        )

    with pytest.raises(ValueError):
        preprocess_ocr_crop(
            crop,
            vertical_padding=-1,
        )


def test_preprocess_rejects_invalid_input() -> None:
    with pytest.raises(TypeError):
        preprocess_ocr_crop("not an image")  # type: ignore[arg-type]
