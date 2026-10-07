"""Synthetic OCR text-line generator for the supported recognition fields.

Each sample is a single rendered text line:

1. field text is sampled in the field's validated format;
2. the text is drawn in a dark ink on a light, variable grey background;
3. a mild rotation is applied and the image is cropped to the text;
4. mild Gaussian blur and Gaussian noise are added;
5. the crop goes through :func:`preprocess_ocr_crop`, the same
   normalization real detector crops receive (grayscale, aspect-preserving
   resize to ``target_height``, padding).

Generation is deterministic for a given seed and font file.

The font is never hardcoded: pass ``font_path`` explicitly, set the
``IDOCR_OCR_FONT`` environment variable, or rely on the first existing
entry of :data:`FALLBACK_FONT_PATHS`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from idocr.data.preprocessing.ocr import (
    DEFAULT_HORIZONTAL_PADDING,
    DEFAULT_TARGET_HEIGHT,
    DEFAULT_VERTICAL_PADDING,
    preprocess_ocr_crop,
)

SUPPORTED_FIELDS: tuple[str, ...] = (
    "name",
    "fathers_name",
    "date_of_birth",
    "gender",
    "aadhaar_number",
    "pan_number",
)

FONT_ENV_VAR = "IDOCR_OCR_FONT"

# Tried in order when no font path is given. Linux/Colab first.
FALLBACK_FONT_PATHS: tuple[str, ...] = (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"

_MALE_FIRST_NAMES = (
    "RAHUL", "AMIT", "SURESH", "RAJESH", "VIKRAM", "ANIL", "SANJAY", "ARJUN",
    "KARAN", "ROHIT", "MANOJ", "DEEPAK", "VIJAY", "ASHOK", "RAVI", "NITIN",
    "SUNIL", "PRAKASH", "MOHAN", "RAMESH", "ADITYA", "HARISH", "GOPAL", "KIRAN",
)
_FEMALE_FIRST_NAMES = (
    "POOJA", "PRIYA", "NEHA", "ANJALI", "SUNITA", "KAVITA", "DIVYA", "MEERA",
    "REKHA", "ANITA", "SNEHA", "SWATI", "LAKSHMI", "GEETA", "NISHA", "RITU",
    "SHALINI", "ASHA", "SEEMA", "PREETI", "ANUSHKA", "DEEPA", "KOMAL", "USHA",
)
_SURNAMES = (
    "JOSHI", "PATEL", "SHARMA", "VERMA", "GUPTA", "SINGH", "KUMAR", "REDDY",
    "NAIR", "IYER", "MEHTA", "SHAH", "DESAI", "KULKARNI", "PATIL", "JADHAV",
    "YADAV", "MISHRA", "PANDEY", "CHAUHAN", "RAO", "PILLAI", "BOSE", "DAS",
)

_DOB_START = date(1950, 1, 1)
_DOB_END = date(2007, 12, 31)


def resolve_font_path(font_path: str | os.PathLike[str] | None = None) -> Path:
    """Return the font file to render with.

    Order: explicit ``font_path``, then ``$IDOCR_OCR_FONT``, then the
    first existing :data:`FALLBACK_FONT_PATHS` entry.

    Raises:
        FileNotFoundError: if an explicit/env path does not exist, or no
            fallback font is installed.
    """
    explicit = font_path if font_path is not None else os.environ.get(FONT_ENV_VAR)
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"font file not found: {path}")
        return path

    for candidate in FALLBACK_FONT_PATHS:
        path = Path(candidate)
        if path.is_file():
            return path

    raise FileNotFoundError(
        "no OCR font found; pass font_path or set "
        f"{FONT_ENV_VAR} (tried: {', '.join(FALLBACK_FONT_PATHS)})"
    )


# ---------------------------------------------------------------------------
# Field text
# ---------------------------------------------------------------------------


def _person_name(rng: random.Random, first_names: tuple[str, ...]) -> str:
    return f"{rng.choice(first_names)} {rng.choice(_SURNAMES)}"


def generate_pan_number(rng: random.Random, surname: str | None = None) -> str:
    """Sample an individual-holder PAN.

    Structure: 3 random letters, ``P`` (individual holder), the surname's
    initial, 4 random digits, 1 random letter. When ``surname`` is not
    given, one is drawn from the synthetic surname list.
    """
    if surname is None:
        surname = rng.choice(_SURNAMES)
    initial = surname.strip()[:1].upper()
    if not initial or initial not in _UPPERCASE:
        raise ValueError(f"surname must start with a Latin letter: {surname!r}")

    return (
        "".join(rng.choice(_UPPERCASE) for _ in range(3))
        + "P"
        + initial
        + "".join(rng.choice(_DIGITS) for _ in range(4))
        + rng.choice(_UPPERCASE)
    )


def generate_field_text(field_name: str, rng: random.Random) -> str:
    """Sample one transcription in the validated format for ``field_name``.

    Formats:
        name, fathers_name: uppercase Latin letters and spaces
        date_of_birth: DD/MM/YYYY
        gender: MALE or FEMALE
        aadhaar_number: XXXX XXXX XXXX (first digit 2-9)
        pan_number: AAAP + surname initial + 4 digits + 1 letter (individual)
    """
    if field_name == "name":
        first_names = rng.choice((_MALE_FIRST_NAMES, _FEMALE_FIRST_NAMES))
        return _person_name(rng, first_names)

    if field_name == "fathers_name":
        return _person_name(rng, _MALE_FIRST_NAMES)

    if field_name == "date_of_birth":
        span = (_DOB_END - _DOB_START).days
        dob = _DOB_START + timedelta(days=rng.randint(0, span))
        return dob.strftime("%d/%m/%Y")

    if field_name == "gender":
        return rng.choice(("MALE", "FEMALE"))

    if field_name == "aadhaar_number":
        digits = str(rng.randint(2, 9)) + "".join(
            rng.choice(_DIGITS) for _ in range(11)
        )
        return " ".join(digits[i:i + 4] for i in range(0, 12, 4))

    if field_name == "pan_number":
        return generate_pan_number(rng)

    raise ValueError(
        f"unsupported field {field_name!r}; expected one of {SUPPORTED_FIELDS}"
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorConfig:
    """Rendering and augmentation settings.

    Augmentations are deliberately mild; every range is sampled uniformly.
    """

    font_path: str | None = None
    font_size: int = 64
    target_height: int = DEFAULT_TARGET_HEIGHT
    horizontal_padding: int = DEFAULT_HORIZONTAL_PADDING
    vertical_padding: int = DEFAULT_VERTICAL_PADDING
    background_range: tuple[int, int] = (200, 250)
    ink_range: tuple[int, int] = (0, 60)
    max_rotation_degrees: float = 2.0
    max_blur_radius: float = 0.8
    max_noise_std: float = 6.0
    crop_margin: int = 4


@dataclass(frozen=True)
class SyntheticTextSample:
    """One generated text-line image and its transcription."""

    field_name: str
    text: str
    image: Image.Image = field(repr=False)


class SyntheticTextGenerator:
    """Deterministic synthetic text-line generator.

    Args:
        config: rendering settings; defaults to :class:`GeneratorConfig`.
        seed: seed for all text and augmentation randomness.
    """

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.font_path = resolve_font_path(self.config.font_path)
        self._font = ImageFont.truetype(str(self.font_path), self.config.font_size)
        self._rng = random.Random(seed)

    def generate_text(self, field_name: str) -> str:
        return generate_field_text(field_name, self._rng)

    def render(self, text: str) -> Image.Image:
        """Render ``text`` into a preprocessed grayscale ``L`` image."""
        if not text:
            raise ValueError("text must not be empty")

        cfg = self.config
        rng = self._rng
        background = rng.randint(*cfg.background_range)
        ink = rng.randint(*cfg.ink_range)
        angle = rng.uniform(-cfg.max_rotation_degrees, cfg.max_rotation_degrees)
        blur_radius = rng.uniform(0.0, cfg.max_blur_radius)
        noise_std = rng.uniform(0.0, cfg.max_noise_std)
        noise_seed = rng.getrandbits(64)

        # Draw with generous margins so rotation never clips the text.
        left, top, right, bottom = self._font.getbbox(text)
        margin = cfg.font_size
        canvas = Image.new(
            "L",
            (right - left + 2 * margin, bottom - top + 2 * margin),
            background,
        )
        ImageDraw.Draw(canvas).text(
            (margin - left, margin - top), text, font=self._font, fill=ink
        )

        rotated = canvas.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=background,
        )

        # Crop to the inked region (computed before blur/noise).
        pixels = np.asarray(rotated)
        ys, xs = np.nonzero(pixels < (background + ink) / 2)
        m = cfg.crop_margin
        crop_box = (
            max(0, int(xs.min()) - m),
            max(0, int(ys.min()) - m),
            min(rotated.width, int(xs.max()) + 1 + m),
            min(rotated.height, int(ys.max()) + 1 + m),
        )
        crop = rotated.crop(crop_box)

        if blur_radius > 0:
            crop = crop.filter(ImageFilter.GaussianBlur(blur_radius))

        noise = np.random.default_rng(noise_seed).normal(
            0.0, noise_std, size=(crop.height, crop.width)
        )
        noisy = np.clip(np.asarray(crop, dtype=np.float32) + noise, 0, 255)
        crop = Image.fromarray(noisy.astype(np.uint8))

        return preprocess_ocr_crop(
            crop,
            target_height=cfg.target_height,
            horizontal_padding=cfg.horizontal_padding,
            vertical_padding=cfg.vertical_padding,
        )

    def generate(self, field_name: str) -> SyntheticTextSample:
        text = self.generate_text(field_name)
        return SyntheticTextSample(
            field_name=field_name,
            text=text,
            image=self.render(text),
        )


def save_png(image: Image.Image, path: str | os.PathLike[str]) -> Path:
    """Save ``image`` as PNG, creating parent directories."""
    out = Path(path)
    if out.suffix.lower() != ".png":
        raise ValueError(f"expected a .png path, got {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, format="PNG")
    return out
