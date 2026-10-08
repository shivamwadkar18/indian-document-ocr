import dataclasses

import numpy as np
import pytest
from PIL import Image

from idocr.data.augmentation.ocr import (
    OcrAugmentConfig,
    OcrLineAugmenter,
    augment_line_image,
    build_ocr_augmentation,
)
from idocr.data.preprocessing.ocr import (
    DEFAULT_HORIZONTAL_PADDING,
    DEFAULT_TARGET_HEIGHT,
    DEFAULT_VERTICAL_PADDING,
)
from idocr.data.recognition.generator import (
    SUPPORTED_FIELDS,
    GeneratorConfig,
    SyntheticTextGenerator,
    resolve_font_path,
)

HEIGHT = DEFAULT_TARGET_HEIGHT + 2 * DEFAULT_VERTICAL_PADDING
ENABLED = OcrAugmentConfig(enabled=True)
ALL_ON = dataclasses.replace(
    ENABLED,
    **{f.name: 1.0 for f in dataclasses.fields(OcrAugmentConfig) if f.name.startswith("p_")},
)


@pytest.fixture(scope="module")
def clean_samples() -> list[tuple[str, Image.Image]]:
    try:
        font = str(resolve_font_path())
    except FileNotFoundError:
        pytest.skip("no OCR font installed")
    gen = SyntheticTextGenerator(GeneratorConfig(font_path=font), seed=3)
    return [(s.text, s.image) for s in (gen.generate(f) for f in SUPPORTED_FIELDS)]


def _assert_valid(image: Image.Image, text: str) -> None:
    pixels = np.asarray(image)
    assert image.mode == "L"
    assert image.height == HEIGHT
    # Enough CTC frames (width // 4) for the label, even with every char repeated.
    assert image.width // 4 >= 2 * len(text) + 1
    # preprocess_ocr_crop padding is intact.
    assert (pixels[:DEFAULT_VERTICAL_PADDING] == 255).all()
    assert (pixels[:, :DEFAULT_HORIZONTAL_PADDING] == 255).all()
    assert (pixels[:, -DEFAULT_HORIZONTAL_PADDING:] == 255).all()
    # Text is still visible: ink is clearly darker than the background.
    inner = pixels[DEFAULT_VERTICAL_PADDING:-DEFAULT_VERTICAL_PADDING,
                   DEFAULT_HORIZONTAL_PADDING:-DEFAULT_HORIZONTAL_PADDING].astype(int)
    assert np.percentile(inner, 95) - np.percentile(inner, 2) >= 40


def test_disabled_is_identity(clean_samples) -> None:
    _, image = clean_samples[0]
    assert OcrLineAugmenter(OcrAugmentConfig())(image, np.random.default_rng(0)) is image
    assert OcrAugmentConfig().enabled is False


def test_deterministic_for_same_seed(clean_samples) -> None:
    for _, image in clean_samples:
        a = augment_line_image(image, ENABLED, seed=[7, 1, 3])
        b = augment_line_image(image, ENABLED, seed=[7, 1, 3])
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_different_seeds_differ(clean_samples) -> None:
    _, image = clean_samples[0]
    outputs = {np.asarray(augment_line_image(image, ENABLED, seed=s)).tobytes() for s in range(8)}
    assert len(outputs) > 1


def test_input_not_modified(clean_samples) -> None:
    _, image = clean_samples[1]
    before = np.asarray(image).copy()
    augment_line_image(image, ALL_ON, seed=0)
    assert np.array_equal(np.asarray(image), before)


@pytest.mark.parametrize("config", [ENABLED, ALL_ON], ids=["default", "all_steps"])
def test_output_valid_for_all_fields(clean_samples, config) -> None:
    for text, image in clean_samples:
        for seed in range(15):
            _assert_valid(augment_line_image(image, config, seed=seed), text)


def test_each_step_returns_uint8_2d(clean_samples) -> None:
    _, image = clean_samples[4]
    inner = np.asarray(image)[DEFAULT_VERTICAL_PADDING:-DEFAULT_VERTICAL_PADDING,
                              DEFAULT_HORIZONTAL_PADDING:-DEFAULT_HORIZONTAL_PADDING].copy()
    for step in build_ocr_augmentation(ALL_ON).steps:
        out = step(inner, np.random.default_rng(1))
        assert out.dtype == np.uint8 and out.ndim == 2 and min(out.shape) >= 8, type(step).__name__


def test_contrast_stays_readable(clean_samples) -> None:
    config = dataclasses.replace(ENABLED, p_contrast=1.0)
    _, image = clean_samples[0]
    for seed in range(30):
        _assert_valid(augment_line_image(image, config, seed=seed), "RAHUL JOSHI")


def test_config_validation_and_mapping() -> None:
    cfg = OcrAugmentConfig.from_mapping({"enabled": True, "hscale_range": [0.9, 1.1]})
    assert cfg.enabled and cfg.hscale_range == (0.9, 1.1)
    assert OcrAugmentConfig.from_mapping(None) == OcrAugmentConfig()
    with pytest.raises(ValueError):
        OcrAugmentConfig.from_mapping({"p_blurr": 0.5})
    with pytest.raises(ValueError):
        OcrAugmentConfig(p_jpeg=1.5)
    with pytest.raises(ValueError):
        OcrAugmentConfig(hscale_range=(1.2, 0.8))
    with pytest.raises(ValueError):
        OcrAugmentConfig(downsample_height_range=(4, 20))


# --- training integration ----------------------------------------------------


def test_dataset_augmentation_per_epoch(tmp_path, clean_samples) -> None:
    pytest.importorskip("torch")
    import json

    from idocr.models.recognizer import Vocabulary
    from idocr.training.recognition.data import RecognitionDataset

    split = tmp_path / "train"
    (split / "images").mkdir(parents=True)
    with (split / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for i, (text, image) in enumerate(clean_samples):
            image.save(split / "images" / f"{i}.png")
            f.write(json.dumps({"image_path": f"images/{i}.png", "text": text, "field_name": "name",
                                "document_type": "pan", "source": "synthetic", "split": "train"}) + "\n")

    ds = RecognitionDataset(split, Vocabulary(), image_height=HEIGHT,
                            augment=OcrLineAugmenter(ENABLED), seed=42)
    plain = RecognitionDataset(split, Vocabulary(), image_height=HEIGHT)

    ds.set_epoch(1)
    first = [np.asarray(ds[i][0]) for i in range(len(ds))]
    again = [np.asarray(ds[i][0]) for i in range(len(ds))]
    ds.set_epoch(2)
    second = [np.asarray(ds[i][0]) for i in range(len(ds))]

    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert any(not np.array_equal(a, b) for a, b in zip(first, second))
    assert np.array_equal(np.asarray(plain[0][0]), np.asarray(clean_samples[0][1]))


def test_train_config_augmentation_block() -> None:
    pytest.importorskip("torch")
    from idocr.training.recognition import RecognizerConfigError, config_from_dict

    cfg = config_from_dict({"experiment_name": "x", "augmentation": {"enabled": True, "p_jpeg": 0.2}})
    assert cfg.augment_config.enabled and cfg.augment_config.p_jpeg == 0.2
    assert config_from_dict({"experiment_name": "x"}).augment_config.enabled is False
    with pytest.raises(RecognizerConfigError, match="augmentation"):
        config_from_dict({"experiment_name": "x", "augmentation": {"p_jpg": 0.2}})
