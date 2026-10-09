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

import hashlib

_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"

MALE_FIRST_NAMES: tuple[str, ...] = (
    "AAKASH", "AARAV", "ABDUL", "ABHINAV", "ABHISHEK", "ACHYUT", "ADITYA", "AJAY", "AJIT", "AKASH",
    "AKHIL", "ALOK", "AMAN", "AMAR", "AMIT", "ANAND", "ANIL", "ANKIT", "ANMOL", "ANURAG",
    "ARIF", "ARJUN", "ARUN", "ARVIND", "ASHISH", "ASHOK", "ASHWIN", "AVINASH", "AYUSH", "BALRAM",
    "BHARAT", "BHASKAR", "BHAVIN", "BHUVAN", "BIKRAM", "BIMAL", "CHANDAN", "CHETAN", "CHIRAG", "DARSHAN",
    "DEEPAK", "DEV", "DEVENDRA", "DHANANJAY", "DHARAM", "DHARMENDRA", "DHIRAJ", "DILIP", "DINESH", "DIVYESH",
    "ESHWAR", "FAISAL", "FARHAN", "GANESH", "GAURAV", "GAUTAM", "GIRISH", "GOPAL", "GOVIND", "GULSHAN",
    "GURPREET", "HARIKRISHNAN", "HARISH", "HARPREET", "HARSH", "HEMANT", "HITESH", "INDRAJIT", "IQBAL", "ISHAN",
    "JAGDISH", "JAIDEEP", "JASWANT", "JATIN", "JAYANT", "JITENDRA", "KAMAL", "KAPIL", "KARAN", "KARTIK",
    "KESHAV", "KIRAN", "KISHORE", "KULDEEP", "KUNAL", "LAKSHMAN", "LALIT", "LOKESH", "MADHAV", "MAHESH",
    "MANISH", "MANOJ", "MAYANK", "MOHAMMED", "MOHAN", "MOHIT", "MUKESH", "NARESH", "NAVEEN", "NEERAJ",
    "NIKHIL", "NITESH", "NITIN", "OM", "OMKAR", "PANKAJ", "PARAS", "PARTH", "PAVAN", "PRABHAT",
    "PRADEEP", "PRAKASH", "PRANAV", "PRASHANT", "PRATEEK", "PRAVEEN", "PUNIT", "PURUSHOTHAMAN", "RAGHAV", "RAHUL",
    "RAJ", "RAJAT", "RAJEEV", "RAJENDRA", "RAJESH", "RAKESH", "RAM", "RAMAN", "RAMESH", "RANJEET",
    "RAVI", "RAVINDRA", "RISHABH", "RITESH", "ROHAN", "ROHIT", "SACHIN", "SAGAR", "SAMIR", "SANDEEP",
    "SANJAY", "SANJEEV", "SANTOSH", "SARVESH", "SATISH", "SAURABH", "SHAILESH", "SHASHANK", "SHEKHAR", "SHIV",
    "SHUBHAM", "SIDDHARTH", "SOHAN", "SOURAV", "SUBHASH", "SUDHIR", "SUJIT", "SUMIT", "SUNDAR", "SUNIL",
    "SURAJ", "SURENDRA", "SURESH", "TARUN", "TEJAS", "TUSHAR", "UDAY", "UMESH", "UTKARSH", "VARUN",
    "VED", "VIJAY", "VIKAS", "VIKRAM", "VIMAL", "VINAY", "VINOD", "VIPIN", "VIPUL", "VISHAL",
    "VISHNU", "VIVEK", "YASH", "YOGESH", "ZAFAR",
)

FEMALE_FIRST_NAMES: tuple[str, ...] = (
    "AANCHAL", "AARUSHI", "AARTI", "AASHNA", "AAYUSHI", "ABHA", "ADITI", "AISHWARYA", "AKANKSHA", "AKSHITA",
    "ALKA", "AMITA", "AMRITA", "ANANYA", "ANITA", "ANJALI", "ANJU", "ANKITA", "ANNAPURNA", "ANSHIKA",
    "ANUPAMA", "ANURADHA", "ANUSHA", "ANUSHKA", "APARNA", "ARCHANA", "ARUNA", "ASHA", "ASHWINI", "AVANI",
    "BARKHA", "BHAGYASHREE", "BHARTI", "BHAVANA", "BHAVNA", "BHAWNA", "BHUMIKA", "BINDUSHA", "CHAITALI", "CHANDANA",
    "CHANDANI", "CHHAVI", "CHITRA", "DAMINI", "DARSHANA", "DEEPA", "DEEPIKA", "DEEPTI", "DEVASREE", "DIVYA",
    "DRISHTI", "EKTA", "ESHA", "FALGUNI", "FARIDA", "FATIMA", "GARIMA", "GAYATRI", "GEETA", "GEETANJALI",
    "GITA", "GUNJAN", "HARINI", "HARPREET", "HARSHA", "HEMA", "HIMANI", "INDU", "INDIRA", "ISHA",
    "ISHITA", "JAYA", "JAYASHREE", "JYOTI", "KAJAL", "KALPANA", "KALYANI", "KAMINI", "KANCHAN", "KARISHMA",
    "KAVITA", "KHUSHBOO", "KIRAN", "KIRTI", "KOMAL", "KRITIKA", "KUSUM", "LAKSHMI", "LALITA", "LATA",
    "LAVANYA", "LEELA", "MADHURI", "MAHIMA", "MAMTA", "MANISHA", "MANJU", "MANPREET", "MAYA", "MEENA",
    "MEENAKSHI", "MEERA", "MEGHA", "MONIKA", "MRIDULA", "MUKTA", "NAINA", "NANDINI", "NEELAM", "NEELIMA",
    "NEHA", "NIDHI", "NIKITA", "NISHA", "NITA", "NUPUR", "PADMA", "PADMAVATHI", "PALLAVI", "PARUL",
    "PAYAL", "POOJA", "POONAM", "PRACHI", "PRAGATI", "PRANITA", "PRATIBHA", "PRATIMA", "PREETI", "PRERNA",
    "PRIYA", "PRIYANKA", "PUSHPA", "RACHANA", "RADHA", "RADHIKA", "RAJNI", "RAKHI", "RASHMI", "REENA",
    "REKHA", "RENUKA", "RICHA", "RINKU", "RITA", "RITU", "RIYA", "ROHINI", "ROOPA", "RUCHI",
    "RUCHIKA", "RUPA", "RUPALI", "SAKSHI", "SANDHYA", "SANGEETA", "SANJANA", "SAPNA", "SARITA", "SAROJ",
    "SAROJINI", "SEEMA", "SHALINI", "SHASHI", "SHEELA", "SHIKHA", "SHILPA", "SHIVANI", "SHOBHA", "SHREEYA",
    "SHREYA", "SHRUTI", "SHWETA", "SIMRAN", "SMRITI", "SNEHA", "SNEHAL", "SONAL", "SONAM", "SONIA",
    "SUCHITRA", "SUDHA", "SUJATA", "SULOCHANA", "SUMAN", "SUMITRA", "SUNITA", "SUPRIYA", "SURABHI", "SUREKHA",
    "SUSHILA", "SUSHMA", "SWATI", "SWETA", "TANISHA", "TANUJA", "TANVI", "TANYA", "TARUNA", "TRIPTI",
    "TRISHA", "TULSI", "UMA", "URMILA", "URVASHI", "USHA", "VAISHALI", "VANDANA", "VARSHA", "VEENA",
    "VIDYA", "VINITA", "YAMINI", "YASHODA", "YOGITA", "ZOYA",
)

SURNAMES: tuple[str, ...] = (
    "ACHARYA", "ADHIKARI", "AGARWAL", "AGNIHOTRI", "AHUJA", "AMBEDKAR", "ANAND", "APTE", "BAGCHI", "BAJAJ",
    "BAKSHI", "BALAKRISHNAN", "BANDYOPADHYAY", "BANERJEE", "BANSAL", "BARMAN", "BASU", "BATRA", "BHADURI", "BHAGAT",
    "BHALLA", "BHARDWAJ", "BHARGAVA", "BHAT", "BHATIA", "BHATTACHARYA", "BHOSALE", "BISWAS", "BORA", "BORKAR",
    "BOSE", "CHADHA", "CHAKRABORTY", "CHANDRA", "CHATTERJEE", "CHATTOPADHYAY", "CHAUHAN", "CHAVAN", "CHAWLA", "CHOPRA",
    "CHOUDHARY", "CHOWDHURY", "DALAL", "DAMODARAN", "DAS", "DASGUPTA", "DATTA", "DAVE", "DEOKAR", "DESAI",
    "DESHMUKH", "DESHPANDE", "DEWAN", "DHAR", "DHILLON", "DIXIT", "DODDA", "DUBEY", "DUTTA", "FERNANDES",
    "GADE", "GAIKWAD", "GANDHI", "GANGULY", "GARG", "GHATAK", "GHOSH", "GILL", "GOGOI", "GOKHALE",
    "GOSWAMI", "GOVINDARAJAN", "GROVER", "GUHA", "GULATI", "GUPTA", "HALDER", "HANDA", "HARIDAS", "HEGDE",
    "HOODA", "IYENGAR", "IYER", "JADHAV", "JAIN", "JAISWAL", "JAYARAMAN", "JHA", "JINDAL", "JOGLEKAR",
    "JOSHI", "KAKKAR", "KALITA", "KAMATH", "KAPOOR", "KAPUR", "KAR", "KASHYAP", "KAUL", "KAUR",
    "KHAN", "KHANNA", "KHATRI", "KHURANA", "KOHLI", "KOTIAN", "KRISHNAN", "KULKARNI", "KUMAR", "KUMARI",
    "LAHOTI", "LAL", "LELE", "LODHA", "MADHAVAN", "MAHAJAN", "MAITI", "MAJUMDAR", "MALHOTRA", "MALIK",
    "MALLICK", "MANDAL", "MANI", "MANOHAR", "MARATHE", "MATHUR", "MAZUMDAR", "MEDHI", "MEHRA", "MEHTA",
    "MENON", "MISHRA", "MITRA", "MITTAL", "MOHANTY", "MUKHERJEE", "MUKHOPADHYAY", "MURTHY", "NADAR", "NAIDU",
    "NAIK", "NAIR", "NAMBIAR", "NANDA", "NARAYAN", "NARAYANAN", "NATARAJAN", "NATH", "NAYAK", "NEGI",
    "NIGAM", "OBEROI", "OJHA", "PAI", "PAL", "PANDEY", "PANDIT", "PANICKER", "PANT", "PARIKH",
    "PARMAR", "PASWAN", "PATEL", "PATHAK", "PATIL", "PATNAIK", "PAUL", "PILLAI", "POOJARY", "PRABHU",
    "PRADHAN", "PRASAD", "PUJARI", "PURI", "QURESHI", "RADHAKRISHNAN", "RAGHAVAN", "RAI", "RAINA", "RAJAGOPAL",
    "RAJAN", "RAJPUT", "RAMACHANDRAN", "RAMAKRISHNAN", "RAMAN", "RAMASWAMY", "RAMESH", "RANA", "RANDHAWA", "RANGANATHAN",
    "RAO", "RASTOGI", "RATHORE", "RAUT", "RAWAT", "RAY", "REDDY", "ROY", "SACHDEV", "SAHA",
    "SAHAY", "SAHNI", "SAHOO", "SAINI", "SAMANT", "SANDHU", "SANE", "SANGHVI", "SANYAL", "SARAF",
    "SARAN", "SARASWAT", "SARKAR", "SARMA", "SAXENA", "SEHGAL", "SEN", "SENGUPTA", "SETH", "SETHI",
    "SHAH", "SHARMA", "SHENOY", "SHETTY", "SHINDE", "SHRIVASTAVA", "SHUKLA", "SINGH", "SINGHAL", "SINGHANIA",
    "SINHA", "SODHI", "SOLANKI", "SOMANI", "SONI", "SRIDHAR", "SRINIVAS", "SRINIVASAN", "SRIVASTAVA", "SUBRAMANIAN",
    "SUNDARAM", "SURI", "SURVE", "SWAMINATHAN", "TALWAR", "TAMBE", "TANDON", "TELANG", "THACKER", "THAKUR",
    "THOMAS", "THOMSON", "THOTA", "TIWARI", "TRIPATHI", "TRIVEDI", "UPADHYAY", "UPPAL", "VAIDYA", "VAISH",
    "VARGHESE", "VARMA", "VARSHNEY", "VASUDEVAN", "VENKATACHALAM", "VENKATARAMAN", "VENKATESH", "VENUGOPAL", "VERMA",
    "VIJAY", "VIRANI", "VOHRA", "VYAS", "WADEKAR", "WADHWA", "WARRIER", "XAVIER", "YADAV", "ZACHARIA", "ZAIDI",
)

_MALE_FIRST_NAMES = MALE_FIRST_NAMES
_FEMALE_FIRST_NAMES = FEMALE_FIRST_NAMES
_SURNAMES = SURNAMES


def full_name_split(first_name: str, surname: str) -> str:
    """Deterministically partition complete name combinations across splits."""
    key = f"{first_name.strip().upper()} {surname.strip().upper()}".encode("utf-8")
    h = int(hashlib.sha256(key).hexdigest()[:8], 16) % 1000
    if h < 800:
        return "train"
    elif h < 900:
        return "valid"
    else:
        return "test"


SPLIT_MALE_NAME_COMBINATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    s: tuple(
        (f, sur)
        for f in MALE_FIRST_NAMES
        for sur in SURNAMES
        if full_name_split(f, sur) == s
    )
    for s in ("train", "valid", "test")
}

SPLIT_FEMALE_NAME_COMBINATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    s: tuple(
        (f, sur)
        for f in FEMALE_FIRST_NAMES
        for sur in SURNAMES
        if full_name_split(f, sur) == s
    )
    for s in ("train", "valid", "test")
}

SPLIT_NAME_COMBINATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    s: SPLIT_MALE_NAME_COMBINATIONS[s] + SPLIT_FEMALE_NAME_COMBINATIONS[s]
    for s in ("train", "valid", "test")
}

# Backward compatibility alias for single-component queries
SPLIT_MALE_FIRST_NAMES: dict[str, tuple[str, ...]] = {
    s: tuple(sorted({f for f, _ in SPLIT_MALE_NAME_COMBINATIONS[s]}))
    for s in ("train", "valid", "test")
}
SPLIT_FEMALE_FIRST_NAMES: dict[str, tuple[str, ...]] = {
    s: tuple(sorted({f for f, _ in SPLIT_FEMALE_NAME_COMBINATIONS[s]}))
    for s in ("train", "valid", "test")
}
SPLIT_SURNAMES: dict[str, tuple[str, ...]] = {
    s: tuple(sorted({sur for _, sur in SPLIT_NAME_COMBINATIONS[s]}))
    for s in ("train", "valid", "test")
}

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


def _apply_case_variation(text: str, rng: random.Random) -> str:
    """Apply realistic case variation (uppercase, title-case, lowercase, or mixed-case)."""
    case_style = rng.choice(("upper", "title", "lower", "mixed"))
    if case_style == "upper":
        return text.upper()
    if case_style == "title":
        return text.title()
    if case_style == "lower":
        return text.lower()
    return "".join(
        c.upper() if rng.random() < 0.5 else c.lower() if c.isalpha() else c
        for c in text
    )


def _person_name(
    rng: random.Random,
    first_names: tuple[str, ...] | None = None,
    surnames: tuple[str, ...] | None = None,
    split: str = "train",
) -> str:
    if first_names is not None and surnames is not None:
        return f"{rng.choice(first_names)} {rng.choice(surnames)}"
    combinations = SPLIT_NAME_COMBINATIONS.get(split, SPLIT_NAME_COMBINATIONS["train"])
    first, sur = rng.choice(combinations)
    return f"{first} {sur}"


def generate_pan_number(
    rng: random.Random,
    surname: str | None = None,
    split: str = "train",
) -> str:
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


def generate_field_text(
    field_name: str,
    rng: random.Random,
    split: str = "train",
) -> str:
    """Sample one transcription in the validated format for ``field_name`` and ``split``.

    Formats:
        name, fathers_name: Latin letters (uppercase/title-case/mixed-case) and spaces
        date_of_birth: DD/MM/YYYY
        gender: MALE or FEMALE (uppercase/title-case/mixed-case)
        aadhaar_number: XXXX XXXX XXXX (first digit 2-9)
        pan_number: AAAP + surname initial + 4 digits + 1 letter (individual, uppercase)
    """
    if field_name == "name":
        combinations = SPLIT_NAME_COMBINATIONS.get(split, SPLIT_NAME_COMBINATIONS["train"])
        first, sur = rng.choice(combinations)
        return _apply_case_variation(f"{first} {sur}", rng)

    if field_name == "fathers_name":
        combinations = SPLIT_MALE_NAME_COMBINATIONS.get(split, SPLIT_MALE_NAME_COMBINATIONS["train"])
        first, sur = rng.choice(combinations)
        return _apply_case_variation(f"{first} {sur}", rng)

    if field_name == "date_of_birth":
        span = (_DOB_END - _DOB_START).days
        dob = _DOB_START + timedelta(days=rng.randint(0, span))
        return dob.strftime("%d/%m/%Y")

    if field_name == "gender":
        gender = rng.choice(("MALE", "FEMALE"))
        return _apply_case_variation(gender, rng)

    if field_name == "aadhaar_number":
        digits = str(rng.randint(2, 9)) + "".join(
            rng.choice(_DIGITS) for _ in range(11)
        )
        return " ".join(digits[i:i + 4] for i in range(0, 12, 4))

    if field_name == "pan_number":
        return generate_pan_number(rng, split=split)

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

    def generate_text(self, field_name: str, split: str = "train") -> str:
        return generate_field_text(field_name, self._rng, split=split)

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

    def generate(self, field_name: str, split: str = "train") -> SyntheticTextSample:
        text = self.generate_text(field_name, split=split)
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
