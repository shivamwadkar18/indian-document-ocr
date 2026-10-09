import random
import re

import numpy as np
import pytest
from PIL import Image

from idocr.data.preprocessing.ocr import (
    DEFAULT_HORIZONTAL_PADDING,
    DEFAULT_TARGET_HEIGHT,
    DEFAULT_VERTICAL_PADDING,
)
from idocr.data.recognition.generator import (
    FONT_ENV_VAR,
    SUPPORTED_FIELDS,
    GeneratorConfig,
    SyntheticTextGenerator,
    _SURNAMES,
    generate_field_text,
    generate_pan_number,
    resolve_font_path,
    save_png,
)

FIELD_PATTERNS = {
    "name": r"[A-Za-z]+( [A-Za-z]+)+",
    "fathers_name": r"[A-Za-z]+( [A-Za-z]+)+",
    "date_of_birth": r"\d{2}/\d{2}/\d{4}",
    "gender": r"(?i:MALE|FEMALE)",
    "aadhaar_number": r"[2-9]\d{3} \d{4} \d{4}",
    "pan_number": r"[A-Z]{3}P[A-Z]\d{4}[A-Z]",
}


@pytest.fixture(scope="module")
def font_path() -> str:
    try:
        return str(resolve_font_path())
    except FileNotFoundError:
        pytest.skip("no OCR font installed")


@pytest.fixture
def generator(font_path: str) -> SyntheticTextGenerator:
    return SyntheticTextGenerator(GeneratorConfig(font_path=font_path), seed=7)


def test_supported_fields() -> None:
    assert set(SUPPORTED_FIELDS) == set(FIELD_PATTERNS)


@pytest.mark.parametrize("field_name", SUPPORTED_FIELDS)
def test_field_text_matches_format(field_name: str) -> None:
    rng = random.Random(0)
    for _ in range(200):
        text = generate_field_text(field_name, rng)
        assert text
        assert re.fullmatch(FIELD_PATTERNS[field_name], text), text


def test_date_of_birth_is_valid_date() -> None:
    from datetime import datetime

    rng = random.Random(1)
    for _ in range(200):
        datetime.strptime(generate_field_text("date_of_birth", rng), "%d/%m/%Y")


def test_pan_is_individual_with_surname_initial() -> None:
    initials = {surname[0] for surname in _SURNAMES}
    rng = random.Random(3)
    for _ in range(500):
        pan = generate_field_text("pan_number", rng)
        assert len(pan) == 10
        assert pan[3] == "P"
        assert pan[4] in initials


def test_pan_uses_given_surname_initial() -> None:
    rng = random.Random(0)
    assert generate_pan_number(rng, surname="KULKARNI")[3:5] == "PK"
    assert generate_pan_number(rng, surname="joshi")[3:5] == "PJ"


def test_pan_remaining_characters_randomized() -> None:
    rng = random.Random(5)
    pans = [generate_pan_number(rng, surname="PATEL") for _ in range(200)]
    for position in (0, 1, 2, 5, 6, 7, 8, 9):
        assert len({pan[position] for pan in pans}) > 5


def test_pan_rejects_non_latin_surname() -> None:
    with pytest.raises(ValueError):
        generate_pan_number(random.Random(0), surname="")


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValueError):
        generate_field_text("address", random.Random(0))


def test_text_generation_deterministic() -> None:
    a = [generate_field_text(f, random.Random(42)) for f in SUPPORTED_FIELDS]
    b = [generate_field_text(f, random.Random(42)) for f in SUPPORTED_FIELDS]
    assert a == b


def test_generation_deterministic_with_seed(font_path: str) -> None:
    config = GeneratorConfig(font_path=font_path)
    first = SyntheticTextGenerator(config, seed=123)
    second = SyntheticTextGenerator(config, seed=123)

    for field_name in SUPPORTED_FIELDS:
        a = first.generate(field_name)
        b = second.generate(field_name)
        assert a.text == b.text
        assert np.array_equal(np.asarray(a.image), np.asarray(b.image))


def test_different_seeds_differ(font_path: str) -> None:
    config = GeneratorConfig(font_path=font_path)
    a = SyntheticTextGenerator(config, seed=1).generate("pan_number").text
    b = SyntheticTextGenerator(config, seed=2).generate("pan_number").text
    assert a != b


@pytest.mark.parametrize("field_name", SUPPORTED_FIELDS)
def test_rendered_image_follows_preprocessing(
    generator: SyntheticTextGenerator, field_name: str
) -> None:
    sample = generator.generate(field_name)
    image = sample.image

    assert sample.field_name == field_name
    assert image.mode == "L"
    assert image.height == DEFAULT_TARGET_HEIGHT + 2 * DEFAULT_VERTICAL_PADDING
    assert image.width > image.height  # a text line is wider than tall

    pixels = np.asarray(image)
    # Padding from preprocess_ocr_crop is pure white.
    assert (pixels[:DEFAULT_VERTICAL_PADDING] == 255).all()
    assert (pixels[-DEFAULT_VERTICAL_PADDING:] == 255).all()
    assert (pixels[:, :DEFAULT_HORIZONTAL_PADDING] == 255).all()
    assert (pixels[:, -DEFAULT_HORIZONTAL_PADDING:] == 255).all()


def test_rendered_image_is_dark_text_on_light_background(
    generator: SyntheticTextGenerator,
) -> None:
    pixels = np.asarray(generator.render("RAHUL JOSHI")).astype(int)
    assert np.median(pixels) > 180  # mostly light background
    assert pixels.min() < 100  # contains dark ink


def test_render_rejects_empty_text(generator: SyntheticTextGenerator) -> None:
    with pytest.raises(ValueError):
        generator.render("")


def test_custom_preprocessing_parameters(font_path: str) -> None:
    config = GeneratorConfig(
        font_path=font_path,
        target_height=32,
        horizontal_padding=2,
        vertical_padding=1,
    )
    image = SyntheticTextGenerator(config, seed=0).render("FEMALE")
    assert image.height == 34


def test_explicit_font_path_used(font_path: str) -> None:
    gen = SyntheticTextGenerator(GeneratorConfig(font_path=font_path), seed=0)
    assert str(gen.font_path) == font_path


def test_missing_font_path_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_font_path(tmp_path / "missing.ttf")


def test_font_env_var(font_path: str, monkeypatch) -> None:
    monkeypatch.setenv(FONT_ENV_VAR, font_path)
    assert str(resolve_font_path()) == font_path


def test_save_png_roundtrip(generator: SyntheticTextGenerator, tmp_path) -> None:
    sample = generator.generate("aadhaar_number")
    out = save_png(sample.image, tmp_path / "nested" / "sample.png")

    with Image.open(out) as loaded:
        assert loaded.format == "PNG"
        assert np.array_equal(np.asarray(loaded), np.asarray(sample.image))


def test_save_png_rejects_other_extensions(
    generator: SyntheticTextGenerator, tmp_path
) -> None:
    with pytest.raises(ValueError):
        save_png(generator.render("MALE"), tmp_path / "sample.jpg")


def test_case_variation_generated() -> None:
    rng = random.Random(42)
    names = [generate_field_text("name", rng) for _ in range(100)]
    genders = [generate_field_text("gender", rng) for _ in range(100)]

    has_upper_name = any(n.isupper() for n in names)
    has_mixed_or_lower_name = any(any(c.islower() for c in n) for n in names)
    assert has_upper_name and has_mixed_or_lower_name

    has_upper_gender = any(g.isupper() for g in genders)
    has_mixed_or_lower_gender = any(any(c.islower() for c in g) for g in genders)
    assert has_upper_gender and has_mixed_or_lower_gender


def test_pan_remains_strictly_uppercase() -> None:
    rng = random.Random(42)
    for _ in range(200):
        pan = generate_field_text("pan_number", rng)
        assert pan.isupper()
        assert re.fullmatch(r"[A-Z]{3}P[A-Z]\d{4}[A-Z]", pan)


def test_split_name_pools_are_disjoint() -> None:
    from idocr.data.recognition.generator import (
        SPLIT_FEMALE_FIRST_NAMES,
        SPLIT_MALE_FIRST_NAMES,
        SPLIT_SURNAMES,
    )

    splits = ("train", "valid", "test")
    for s1 in splits:
        for s2 in splits:
            if s1 >= s2:
                continue
            assert not (set(SPLIT_MALE_FIRST_NAMES[s1]) & set(SPLIT_MALE_FIRST_NAMES[s2]))
            assert not (set(SPLIT_FEMALE_FIRST_NAMES[s1]) & set(SPLIT_FEMALE_FIRST_NAMES[s2]))
            assert not (set(SPLIT_SURNAMES[s1]) & set(SPLIT_SURNAMES[s2]))


def test_generated_names_disjoint_across_splits() -> None:
    splits = ("train", "valid", "test")
    names_by_split: dict[str, set[str]] = {}
    for s in splits:
        rng = random.Random(100)
        names = [generate_field_text("name", rng, split=s) for _ in range(500)]
        names_by_split[s] = {re.sub(r"\s+", " ", n.strip().lower()) for n in names}

    assert not (names_by_split["train"] & names_by_split["valid"])
    assert not (names_by_split["train"] & names_by_split["test"])
    assert not (names_by_split["valid"] & names_by_split["test"])


def test_generated_fathers_names_disjoint_across_splits() -> None:
    splits = ("train", "valid", "test")
    names_by_split: dict[str, set[str]] = {}
    for s in splits:
        rng = random.Random(200)
        names = [generate_field_text("fathers_name", rng, split=s) for _ in range(500)]
        names_by_split[s] = {re.sub(r"\s+", " ", n.strip().lower()) for n in names}

    assert not (names_by_split["train"] & names_by_split["valid"])
    assert not (names_by_split["train"] & names_by_split["test"])
    assert not (names_by_split["valid"] & names_by_split["test"])
