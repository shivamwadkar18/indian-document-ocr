import dataclasses
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from idocr.data.recognition import RecognitionSample
from idocr.data.recognition.dataset import (
    ANNOTATIONS_FILENAME,
    INFO_FILENAME,
    SPLITS,
    STAGING_DIRNAME,
    RecognitionDatasetConfig,
    generate_recognition_dataset,
    validate_recognition_dataset,
)
from idocr.data.recognition.generator import (
    SUPPORTED_FIELDS,
    GeneratorConfig,
    resolve_font_path,
)
from idocr.utils.paths import find_project_root

CONFIG_PATH = find_project_root() / "configs" / "recognition" / "recognition_dataset.yaml"
PAN_PATTERN = r"[A-Z]{3}P[A-Z][0-9]{4}[A-Z]"

# 120 samples -> 20 per field -> 16 / 2 / 2.
SMALL_TOTAL = 120
SMALL_EXPECTED = {"train": 16, "valid": 2, "test": 2}


@pytest.fixture(scope="module")
def font_path() -> str:
    try:
        return str(resolve_font_path())
    except FileNotFoundError:
        pytest.skip("no OCR font installed")


@pytest.fixture(scope="module")
def small_config(font_path: str) -> RecognitionDatasetConfig:
    config = RecognitionDatasetConfig.from_yaml(CONFIG_PATH)
    return dataclasses.replace(
        config,
        seed=123,
        total_samples=SMALL_TOTAL,
        generator=dataclasses.replace(config.generator, font_path=font_path),
    )


@pytest.fixture(scope="module")
def dataset_dir(small_config, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("recognition")
    generate_recognition_dataset(small_config, out)
    return out


def _records(dataset_dir: Path, split: str) -> list[dict]:
    lines = (dataset_dir / split / ANNOTATIONS_FILENAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in lines.splitlines()]


# --- configuration ---------------------------------------------------------


def test_project_config_counts() -> None:
    config = RecognitionDatasetConfig.from_yaml(CONFIG_PATH)

    assert config.total_samples == 30000
    assert config.seed == 42
    assert config.records_per_split() == {"train": 4000, "valid": 500, "test": 500}
    assert config.generator.target_height == 48
    assert config.generator.horizontal_padding == 8
    assert config.generator.vertical_padding == 4


def test_counts_must_divide_across_fields() -> None:
    config = RecognitionDatasetConfig.from_yaml(CONFIG_PATH)
    with pytest.raises(ValueError):
        dataclasses.replace(config, total_samples=30001)


def test_counts_must_divide_across_splits() -> None:
    config = RecognitionDatasetConfig.from_yaml(CONFIG_PATH)
    with pytest.raises(ValueError):
        dataclasses.replace(config, total_samples=6 * 15)  # 15 * 0.1 = 1.5


def test_split_ratios_must_sum_to_one() -> None:
    config = RecognitionDatasetConfig.from_yaml(CONFIG_PATH)
    with pytest.raises(ValueError):
        dataclasses.replace(
            config, split_ratios={"train": 0.8, "valid": 0.1, "test": 0.2}
        )


# --- generated dataset -----------------------------------------------------


def test_split_totals(dataset_dir: Path) -> None:
    for split in SPLITS:
        records = _records(dataset_dir, split)
        images = list((dataset_dir / split / "images").glob("*.png"))
        expected = SMALL_EXPECTED[split] * len(SUPPORTED_FIELDS)
        assert len(records) == expected
        assert len(images) == expected


def test_exact_field_balance(dataset_dir: Path) -> None:
    for split in SPLITS:
        counts = {f: 0 for f in SUPPORTED_FIELDS}
        for record in _records(dataset_dir, split):
            counts[record["field_name"]] += 1
        assert counts == {f: SMALL_EXPECTED[split] for f in SUPPORTED_FIELDS}


def test_records_are_valid_samples(dataset_dir: Path) -> None:
    for split in SPLITS:
        for record in _records(dataset_dir, split):
            sample = RecognitionSample(**record)
            assert sample.to_dict() == record
            assert sample.text.strip()
            assert sample.source == "synthetic"
            assert sample.split == split


def test_image_annotation_pairing(dataset_dir: Path, small_config) -> None:
    expected_height = (
        small_config.generator.target_height
        + 2 * small_config.generator.vertical_padding
    )
    for split in SPLITS:
        split_dir = dataset_dir / split
        paths = [r["image_path"] for r in _records(dataset_dir, split)]
        on_disk = {f"images/{p.name}" for p in (split_dir / "images").iterdir()}

        assert len(paths) == len(set(paths))
        assert set(paths) == on_disk
        for rel in paths[:12]:
            with Image.open(split_dir / rel) as image:
                assert image.format == "PNG"
                assert image.mode == "L"
                assert image.height == expected_height


def test_pan_structure_and_surname(dataset_dir: Path) -> None:
    for split in SPLITS:
        records = _records(dataset_dir, split)
        names = {
            r["image_path"].split("_")[0]: r["text"]
            for r in records
            if r["field_name"] == "name"
        }
        for r in records:
            if r["field_name"] != "pan_number":
                continue
            assert re.fullmatch(PAN_PATTERN, r["text"])
            surname = names[r["image_path"].split("_")[0]].split()[-1]
            assert r["text"][4] == surname[0]


def test_document_types(dataset_dir: Path, small_config) -> None:
    for split in SPLITS:
        for r in _records(dataset_dir, split):
            assert r["document_type"] in small_config.document_types[r["field_name"]]


def test_validator_accepts_generated_dataset(dataset_dir: Path, small_config) -> None:
    result = validate_recognition_dataset(small_config, dataset_dir)

    assert result.is_valid, result.errors
    assert result.total_images == SMALL_TOTAL
    assert result.total_annotations == SMALL_TOTAL
    assert result.modes == {"L"}


def test_validator_detects_broken_dataset(small_config, tmp_path: Path) -> None:
    generate_recognition_dataset(small_config, tmp_path)
    next((tmp_path / "test" / "images").glob("*.png")).unlink()
    annotations = tmp_path / "valid" / ANNOTATIONS_FILENAME
    lines = annotations.read_text(encoding="utf-8").splitlines()
    annotations.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")

    result = validate_recognition_dataset(small_config, tmp_path)

    assert not result.is_valid
    assert any("missing image" in e for e in result.errors)
    assert any("images without annotations" in e for e in result.errors)
    assert any("expected 2" in e for e in result.errors)


def test_info_file(dataset_dir: Path, small_config) -> None:
    info = json.loads((dataset_dir / INFO_FILENAME).read_text(encoding="utf-8"))
    assert info["seed"] == 123
    assert info["total_samples"] == SMALL_TOTAL
    assert info["samples_per_field_by_split"] == SMALL_EXPECTED
    assert info["source"] == "synthetic"


# --- determinism and rerun safety ----------------------------------------


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_generation_is_deterministic(small_config, dataset_dir: Path, tmp_path: Path) -> None:
    generate_recognition_dataset(small_config, tmp_path)
    assert _tree_bytes(tmp_path) == _tree_bytes(dataset_dir)


def test_different_seed_changes_dataset(small_config, dataset_dir: Path, tmp_path: Path) -> None:
    other = dataclasses.replace(small_config, seed=124)
    generate_recognition_dataset(other, tmp_path)
    assert _records(tmp_path, "train") != _records(dataset_dir, "train")


def test_refuses_to_mix_with_existing_dataset(small_config, tmp_path: Path) -> None:
    generate_recognition_dataset(small_config, tmp_path)
    before = _tree_bytes(tmp_path)

    with pytest.raises(FileExistsError):
        generate_recognition_dataset(small_config, tmp_path)

    assert _tree_bytes(tmp_path) == before
    assert not (tmp_path / STAGING_DIRNAME).exists()


def test_overwrite_replaces_only_managed_entries(small_config, tmp_path: Path) -> None:
    generate_recognition_dataset(small_config, tmp_path)
    stale = tmp_path / "train" / "images" / "stale.png"
    stale.write_bytes(b"old")
    unrelated = tmp_path / "smoke" / "name.png"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"keep")

    generate_recognition_dataset(small_config, tmp_path, overwrite=True)

    assert not stale.exists()
    assert unrelated.read_bytes() == b"keep"
    assert validate_recognition_dataset(small_config, tmp_path).is_valid
    assert not (tmp_path / STAGING_DIRNAME).exists()


def test_partial_existing_dataset_is_detected(small_config, tmp_path: Path) -> None:
    (tmp_path / "valid").mkdir()
    with pytest.raises(FileExistsError):
        generate_recognition_dataset(small_config, tmp_path)


def test_failed_build_leaves_no_staging(small_config, tmp_path: Path) -> None:
    broken = dataclasses.replace(
        small_config, generator=GeneratorConfig(font_path=str(tmp_path / "x.ttf"))
    )
    with pytest.raises(FileNotFoundError):
        generate_recognition_dataset(broken, tmp_path)
    assert not any(tmp_path.iterdir())
