from pathlib import Path
import os

import numpy as np
import pytest

from idocr.models.detector import UltralyticsFieldDetector
from idocr.types import BBox, TextRegion


def test_text_region_contains_detector_class() -> None:
    region = TextRegion(
        bbox=BBox(10, 20, 100, 120),
        confidence=0.9,
        class_id=4,
        class_name="pan_number",
    )

    result = region.to_dict()

    assert result["class_id"] == 4
    assert result["class_name"] == "pan_number"
    assert result["confidence"] == 0.9


def test_detector_rejects_invalid_image(tmp_path: Path) -> None:
    model_path = os.environ.get("IDOCR_TEST_MODEL")

    if not model_path or not Path(model_path).exists():
        pytest.skip("IDOCR_TEST_MODEL not configured.")

    detector = UltralyticsFieldDetector(
        model_path,
        device="cpu",
    )

    invalid = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError):
        detector.detect(invalid)


def test_detector_returns_field_regions(tmp_path: Path) -> None:
    model_path = os.environ.get("IDOCR_TEST_MODEL")
    image_path = os.environ.get("IDOCR_TEST_IMAGE")

    if (
        not model_path
        or not Path(model_path).exists()
        or not image_path
        or not Path(image_path).exists()
    ):
        pytest.skip("Detector test assets not configured.")

    from PIL import Image

    image = np.asarray(
        Image.open(image_path).convert("RGB"),
        dtype=np.uint8,
    )

    detector = UltralyticsFieldDetector(
        model_path,
        device="cpu",
        imgsz=640,
        conf=0.25,
    )

    regions = detector.detect(image)

    assert regions

    for region in regions:
        assert region.bbox.x1 >= 0
        assert region.bbox.y1 >= 0
        assert region.bbox.x2 >= region.bbox.x1
        assert region.bbox.y2 >= region.bbox.y1
        assert 0.0 <= region.confidence <= 1.0
        assert region.class_id >= 0
        assert region.class_name is not None
