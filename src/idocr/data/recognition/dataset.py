"""Synthetic field-level OCR recognition dataset generation and validation.

Layout written under ``output_dir``::

    train/images/<record>_<field>.png
    train/annotations.jsonl
    valid/...
    test/...
    dataset_info.json

Each JSONL line is a :class:`RecognitionSample`; ``image_path`` is relative
to the split directory (the directory holding ``annotations.jsonl``).

Samples are generated record by record: one synthetic identity yields one
crop per field, so every field is exactly balanced in every split. The PAN
of a record uses the surname of that record's ``name``.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from idocr.data.recognition.generator import (
    SUPPORTED_FIELDS,
    GeneratorConfig,
    SyntheticTextGenerator,
    generate_field_text,
    generate_pan_number,
    save_png,
)
from idocr.data.recognition.schema import RecognitionSample
from idocr.utils.config import load_config

SPLITS: tuple[str, ...] = ("train", "valid", "test")
SOURCE = "synthetic"
ANNOTATIONS_FILENAME = "annotations.jsonl"
IMAGES_DIRNAME = "images"
INFO_FILENAME = "dataset_info.json"
STAGING_DIRNAME = ".staging"

# Everything this module owns inside output_dir. Nothing else is touched.
MANAGED_ENTRIES: tuple[str, ...] = (*SPLITS, INFO_FILENAME)

FIELD_TEXT_PATTERNS: dict[str, str] = {
    "name": r"[A-Z]+( [A-Z]+)+",
    "fathers_name": r"[A-Z]+( [A-Z]+)+",
    "date_of_birth": r"\d{2}/\d{2}/\d{4}",
    "gender": r"MALE|FEMALE",
    "aadhaar_number": r"[2-9]\d{3} \d{4} \d{4}",
    "pan_number": r"[A-Z]{3}P[A-Z]\d{4}[A-Z]",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecognitionDatasetConfig:
    """Settings for one full synthetic dataset build."""

    seed: int
    total_samples: int
    split_ratios: Mapping[str, float]
    document_types: Mapping[str, tuple[str, ...]]
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    output_dir: Path = Path("data/processed/recognition")

    def __post_init__(self) -> None:
        if set(self.document_types) != set(SUPPORTED_FIELDS):
            raise ValueError(
                f"fields must be exactly {SUPPORTED_FIELDS}, "
                f"got {tuple(self.document_types)}"
            )
        for field_name, doc_types in self.document_types.items():
            if not doc_types:
                raise ValueError(f"{field_name}: document_types must not be empty")
        if set(self.split_ratios) != set(SPLITS):
            raise ValueError(f"splits must be exactly {SPLITS}")
        if abs(sum(self.split_ratios.values()) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        self.records_per_split()  # validates the counts

    @property
    def fields(self) -> tuple[str, ...]:
        return SUPPORTED_FIELDS

    def records_per_split(self) -> dict[str, int]:
        """Records (= samples per field) in each split.

        Raises:
            ValueError: if the counts cannot be split exactly.
        """
        if self.total_samples <= 0:
            raise ValueError("total_samples must be positive")
        if self.total_samples % len(self.fields):
            raise ValueError(
                f"total_samples ({self.total_samples}) must be divisible by "
                f"the number of fields ({len(self.fields)})"
            )
        per_field = self.total_samples // len(self.fields)

        counts: dict[str, int] = {}
        for split in SPLITS:
            exact = per_field * self.split_ratios[split]
            if abs(exact - round(exact)) > 1e-6:
                raise ValueError(
                    f"{split}: {per_field} samples per field x "
                    f"{self.split_ratios[split]} is not a whole number"
                )
            counts[split] = round(exact)
        return counts

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RecognitionDatasetConfig":
        generation = data.get("generation", {})
        preprocessing = data.get("preprocessing", {})
        fields = data.get("fields", {})

        generator = GeneratorConfig(
            font_path=generation.get("font_path"),
            target_height=preprocessing["target_height"],
            horizontal_padding=preprocessing["horizontal_padding"],
            vertical_padding=preprocessing["vertical_padding"],
        )
        return cls(
            seed=int(generation["seed"]),
            total_samples=int(generation["total_samples"]),
            split_ratios={k: float(v) for k, v in data["splits"].items()},
            document_types={
                name: tuple(spec["document_types"]) for name, spec in fields.items()
            },
            generator=generator,
            output_dir=Path(generation.get("output_dir", cls.output_dir)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RecognitionDatasetConfig":
        return cls.from_mapping(load_config(path))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _image_filename(record_index: int, field_name: str) -> str:
    return f"{record_index:06d}_{field_name}.png"


def _record_texts(rng: random.Random) -> dict[str, str]:
    """Sample one synthetic identity: one transcription per field."""
    texts = {
        field_name: generate_field_text(field_name, rng)
        for field_name in SUPPORTED_FIELDS
        if field_name != "pan_number"
    }
    surname = texts["name"].split()[-1]
    texts["pan_number"] = generate_pan_number(rng, surname=surname)
    return texts


def existing_managed_entries(output_dir: str | Path) -> list[Path]:
    """Generated-dataset entries already present in ``output_dir``."""
    root = Path(output_dir)
    return [root / name for name in MANAGED_ENTRIES if (root / name).exists()]


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def generate_recognition_dataset(
    config: RecognitionDatasetConfig,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Generate the full dataset and return its ``dataset_info`` mapping.

    The build is written to ``output_dir/.staging`` and moved into place
    only after it completes, so an interrupted run never leaves a partial
    dataset in the split directories.

    Raises:
        FileExistsError: if a previous dataset exists and ``overwrite`` is
            false. With ``overwrite``, only :data:`MANAGED_ENTRIES` are
            removed; other files in ``output_dir`` (e.g. ``smoke/``) are kept.
    """
    root = Path(output_dir) if output_dir is not None else config.output_dir
    existing = existing_managed_entries(root)
    if existing and not overwrite:
        raise FileExistsError(
            "recognition dataset already exists; rerun with overwrite to "
            "replace it: " + ", ".join(str(p) for p in existing)
        )

    records_per_split = config.records_per_split()
    text_rng = random.Random(config.seed)
    renderer = SyntheticTextGenerator(
        config.generator, seed=text_rng.getrandbits(64)
    )

    staging = root / STAGING_DIRNAME
    _remove(staging)  # leftover from an interrupted run; always ours
    staging.mkdir(parents=True)

    total = sum(records_per_split.values()) * len(SUPPORTED_FIELDS)
    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=total, unit="img", desc="generating")

    try:
        for split in SPLITS:
            split_dir = staging / split
            (split_dir / IMAGES_DIRNAME).mkdir(parents=True)
            with (split_dir / ANNOTATIONS_FILENAME).open(
                "w", encoding="utf-8", newline="\n"
            ) as annotations:
                for record_index in range(records_per_split[split]):
                    texts = _record_texts(text_rng)
                    for field_name in SUPPORTED_FIELDS:
                        doc_types = config.document_types[field_name]
                        rel_path = (
                            f"{IMAGES_DIRNAME}/"
                            f"{_image_filename(record_index, field_name)}"
                        )
                        sample = RecognitionSample(
                            image_path=rel_path,
                            text=texts[field_name],
                            field_name=field_name,
                            document_type=doc_types[record_index % len(doc_types)],
                            source=SOURCE,
                            split=split,
                        )
                        save_png(
                            renderer.render(sample.text), split_dir / rel_path
                        )
                        annotations.write(json.dumps(sample.to_dict()) + "\n")
                        if bar is not None:
                            bar.update()

        info = {
            "source": SOURCE,
            "seed": config.seed,
            "total_samples": total,
            "fields": list(SUPPORTED_FIELDS),
            "samples_per_field_by_split": records_per_split,
            "split_ratios": dict(config.split_ratios),
            "document_types": {k: list(v) for k, v in config.document_types.items()},
            "image_path_relative_to": "split directory",
            "font_file": renderer.font_path.name,
            "generator_config": asdict(config.generator),
        }
        (staging / INFO_FILENAME).write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        _remove(staging)
        raise
    finally:
        if bar is not None:
            bar.close()

    for path in existing:
        _remove(path)
    for name in MANAGED_ENTRIES:
        (staging / name).rename(root / name)
    staging.rmdir()
    return info


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class DatasetValidationResult:
    errors: list[str] = field(default_factory=list)
    images_by_split: dict[str, int] = field(default_factory=dict)
    annotations_by_split: dict[str, int] = field(default_factory=dict)
    by_split_and_field: dict[str, dict[str, int]] = field(default_factory=dict)
    by_document_type: dict[str, int] = field(default_factory=dict)
    width_range: tuple[int, int] | None = None
    height_range: tuple[int, int] | None = None
    modes: set[str] = field(default_factory=set)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def total_images(self) -> int:
        return sum(self.images_by_split.values())

    @property
    def total_annotations(self) -> int:
        return sum(self.annotations_by_split.values())


def validate_recognition_dataset(
    config: RecognitionDatasetConfig,
    output_dir: str | Path | None = None,
    *,
    check_images: bool = True,
) -> DatasetValidationResult:
    """Check a generated dataset against ``config``.

    Verifies RecognitionSample conformity, source/split values, text
    formats (incl. PAN structure), exact per-field balance, document types,
    1:1 image/annotation pairing and, with ``check_images``, that every
    image is a grayscale PNG with the configured height.
    """
    root = Path(output_dir) if output_dir is not None else config.output_dir
    result = DatasetValidationResult()
    expected = config.records_per_split()
    gen = config.generator
    expected_height = gen.target_height + 2 * gen.vertical_padding
    widths: list[int] = []
    heights: list[int] = []
    doc_counter: Counter[str] = Counter()

    for split in SPLITS:
        split_dir = root / split
        annotations_path = split_dir / ANNOTATIONS_FILENAME
        image_files = sorted((split_dir / IMAGES_DIRNAME).glob("*"))
        result.images_by_split[split] = len(image_files)

        if not annotations_path.is_file():
            result.errors.append(f"{split}: missing {ANNOTATIONS_FILENAME}")
            continue

        field_counts: Counter[str] = Counter()
        referenced: list[str] = []
        lines = annotations_path.read_text(encoding="utf-8").splitlines()
        result.annotations_by_split[split] = len(lines)

        for line_no, line in enumerate(lines, start=1):
            where = f"{split}/{ANNOTATIONS_FILENAME}:{line_no}"
            try:
                sample = RecognitionSample(**json.loads(line))
            except (ValueError, TypeError) as exc:
                result.errors.append(f"{where}: invalid record ({exc})")
                continue

            if sample.split != split:
                result.errors.append(f"{where}: split {sample.split!r} != {split!r}")
            if sample.source != SOURCE:
                result.errors.append(f"{where}: source {sample.source!r}")
            if sample.field_name not in FIELD_TEXT_PATTERNS:
                result.errors.append(f"{where}: unknown field {sample.field_name!r}")
                continue
            if not sample.text.strip():
                result.errors.append(f"{where}: blank text")
            if not re.fullmatch(FIELD_TEXT_PATTERNS[sample.field_name], sample.text):
                result.errors.append(
                    f"{where}: {sample.field_name} text {sample.text!r} "
                    "does not match the expected format"
                )
            if sample.document_type not in config.document_types[sample.field_name]:
                result.errors.append(
                    f"{where}: document_type {sample.document_type!r} not allowed "
                    f"for {sample.field_name}"
                )

            field_counts[sample.field_name] += 1
            doc_counter[sample.document_type] += 1
            referenced.append(sample.image_path)

            if check_images:
                image_path = split_dir / sample.image_path
                if not image_path.is_file():
                    result.errors.append(f"{where}: missing image {sample.image_path}")
                    continue
                with Image.open(image_path) as image:
                    result.modes.add(image.mode)
                    widths.append(image.width)
                    heights.append(image.height)
                    if image.format != "PNG":
                        result.errors.append(f"{where}: {image.format} is not PNG")
                    if image.mode != "L":
                        result.errors.append(f"{where}: mode {image.mode} is not L")
                    if image.height != expected_height:
                        result.errors.append(
                            f"{where}: height {image.height} != {expected_height}"
                        )

        result.by_split_and_field[split] = {
            f: field_counts.get(f, 0) for f in SUPPORTED_FIELDS
        }
        for field_name in SUPPORTED_FIELDS:
            if field_counts.get(field_name, 0) != expected[split]:
                result.errors.append(
                    f"{split}: {field_name} has {field_counts.get(field_name, 0)} "
                    f"samples, expected {expected[split]}"
                )

        duplicates = [p for p, n in Counter(referenced).items() if n > 1]
        if duplicates:
            result.errors.append(
                f"{split}: {len(duplicates)} images referenced more than once"
            )
        on_disk = {f"{IMAGES_DIRNAME}/{p.name}" for p in image_files}
        orphans = on_disk - set(referenced)
        if orphans:
            result.errors.append(f"{split}: {len(orphans)} images without annotations")
        missing = set(referenced) - on_disk
        if missing and not check_images:  # check_images reports each one
            result.errors.append(f"{split}: {len(missing)} annotated images missing")

    result.by_document_type = dict(sorted(doc_counter.items()))
    if widths:
        result.width_range = (min(widths), max(widths))
        result.height_range = (min(heights), max(heights))
    return result
