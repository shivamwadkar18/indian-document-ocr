"""Field-targeted glyph scale / horizontal spacing augmentation (GlyphScaleSpacing)."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from idocr.data.augmentation.ocr import (
    GlyphScaleSpacing,
    OcrAugmentConfig,
    OcrLineAugmenter,
    augment_line_image,
)
from idocr.data.preprocessing.ocr import (
    DEFAULT_HORIZONTAL_PADDING as HPAD,
    DEFAULT_TARGET_HEIGHT as TARGET,
    DEFAULT_VERTICAL_PADDING as VPAD,
)
from idocr.data.recognition.generator import (
    SUPPORTED_FIELDS,
    GeneratorConfig,
    SyntheticTextGenerator,
    resolve_font_path,
)
from idocr.utils.paths import find_project_root

ROOT = find_project_root()
HEIGHT = TARGET + 2 * VPAD
TARGETED = ("aadhaar_number", "date_of_birth", "name")
BASELINE = OcrAugmentConfig(enabled=True)
GENERIC_OFF = dataclasses.replace(
    BASELINE, **{f.name: 0.0 for f in dataclasses.fields(OcrAugmentConfig) if f.name.startswith("p_")}
)


def glyph_only(**kw) -> OcrAugmentConfig:
    """Only the new step, always applied; generic steps off."""
    return dataclasses.replace(GENERIC_OFF, **{"p_glyph_scale": 1.0, "p_hcompress": 1.0, **kw})


@pytest.fixture(scope="module")
def clean_samples() -> list[tuple[str, str, Image.Image]]:
    """(field_name, text, stored-format image) per field."""
    try:
        font = str(resolve_font_path())
    except FileNotFoundError:
        pytest.skip("no OCR font installed")
    gen = SyntheticTextGenerator(GeneratorConfig(font_path=font), seed=11)
    out = []
    for _ in range(3):
        out += [(f, s.text, s.image) for f, s in ((f, gen.generate(f)) for f in SUPPORTED_FIELDS)]
    return out


def _inner(image: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[VPAD:-VPAD, HPAD:-HPAD])


def _ink_rows(arr: np.ndarray) -> np.ndarray:
    lo, bg = np.percentile(arr, (1, 90))
    return np.flatnonzero((arr < (lo + bg) / 2).any(axis=1))


def _step(cfg: OcrAugmentConfig) -> GlyphScaleSpacing:
    return GlyphScaleSpacing(cfg, target_height=TARGET, vertical_padding=VPAD)


# --- config ------------------------------------------------------------------


def test_off_by_default_and_baseline_untouched() -> None:
    cfg = OcrAugmentConfig()
    assert cfg.p_glyph_scale == 0 and cfg.p_hcompress == 0 and not cfg.glyph_scale_active
    assert OcrLineAugmenter(BASELINE).glyph_step is None


def test_config_mapping_and_validation() -> None:
    cfg = OcrAugmentConfig.from_mapping({"enabled": True, "p_glyph_scale": 0.5,
                                         "glyph_height_range": [0.5, 0.7], "scale_fields": ["name"]})
    assert cfg.glyph_height_range == (0.5, 0.7) and cfg.scale_fields == ("name",)
    assert cfg.glyph_scale_active
    for bad in ({"glyph_height_range": (0.0, 0.5)}, {"glyph_height_range": (0.5, 1.2)},
                {"glyph_height_range": (0.8, 0.5)}, {"hcompress_range": (0.0, 1.0)},
                {"hcompress_range": (0.5, 1.5)}, {"min_char_width": 0.0}, {"p_hcompress": 1.5},
                {"scale_fields": ("aadhaar",)}):
        with pytest.raises(ValueError):
            OcrAugmentConfig(**bad)


def test_experiment_config_differs_from_baseline_only_in_name_and_new_keys() -> None:
    pytest.importorskip("torch")
    from idocr.training.recognition import load_recognizer_config

    base = load_recognizer_config(ROOT / "configs" / "recognition" / "crnn_mixedcase_augmented_t4.yaml")
    exp = load_recognizer_config(ROOT / "configs" / "recognition" / "crnn_glyph_scale_spacing_t4.yaml")
    b, e = base.to_dict(), exp.to_dict()
    assert e.pop("experiment_name") == "crnn_glyph_scale_spacing_t4_10ep"
    assert b.pop("experiment_name") == "crnn_mixedcase_augmented_t4_10ep"
    assert e.pop("augmentation") != b.pop("augmentation")
    assert e == b  # every training parameter unchanged

    new_keys = {"p_glyph_scale", "glyph_height_range", "p_hcompress", "hcompress_range",
                "min_char_width", "scale_fields"}
    assert set(exp.augmentation) == {"enabled"} | new_keys
    generic = [f.name for f in dataclasses.fields(OcrAugmentConfig) if f.name not in new_keys]
    for name in generic:
        assert getattr(exp.augment_config, name) == getattr(base.augment_config, name), name
    assert exp.augment_config.glyph_height_range == (0.45, 0.80)
    assert exp.augment_config.hcompress_range == (0.5, 1.0)
    assert set(exp.augment_config.scale_fields) == set(TARGETED)


# --- bounds ------------------------------------------------------------------


def test_glyph_height_bounds(clean_samples) -> None:
    lo, hi = 0.45, 0.80
    cfg = glyph_only(p_hcompress=0.0, glyph_height_range=(lo, hi), min_char_width=1.0)
    fractions = []
    for field, text, image in clean_samples:
        if field not in TARGETED:
            continue
        for seed in range(10):
            out = np.asarray(augment_line_image(image, cfg, seed, field_name=field, text=text))
            assert out.shape[0] == HEIGHT
            rows = _ink_rows(out)
            fractions.append((rows[-1] - rows[0] + 1) / HEIGHT)
    # Measured with a mid-ink threshold after resampling, hence the tolerance.
    assert min(fractions) >= lo - 0.06 and max(fractions) <= hi + 0.03
    assert min(fractions) < 0.52  # smaller than the synthetic ~0.75 is actually reached


def test_horizontal_compression_bounds(clean_samples) -> None:
    lo, hi = 0.5, 1.0
    cfg = glyph_only(p_glyph_scale=0.0, hcompress_range=(lo, hi), min_char_width=1.0)
    step = _step(cfg)
    ratios = []
    for field, text, image in clean_samples:
        inner = _inner(image)
        for seed in range(10):
            out = step(inner, np.random.default_rng(seed), len(text))
            assert out.shape[0] == inner.shape[0]  # height untouched
            ratios.append(out.shape[1] / inner.shape[1])
    tol = 1 / min(_inner(im).shape[1] for _, _, im in clean_samples)
    assert min(ratios) >= lo - tol and max(ratios) <= hi + tol
    assert min(ratios) < 0.6


def test_min_char_width_keeps_ctc_frames(clean_samples) -> None:
    # Extreme draws: smallest glyphs and strongest compression every time.
    cfg = glyph_only(glyph_height_range=(0.45, 0.45), hcompress_range=(0.5, 0.5), min_char_width=10.0)
    for field, text, image in clean_samples:
        if field not in TARGETED:
            continue
        out = augment_line_image(image, cfg, 0, field_name=field, text=text)
        inner_w = out.width - 2 * HPAD
        assert inner_w / len(text) >= 10.0 - 0.5
        assert out.width // 4 >= 2 * len(text) + 1


# --- content preservation ----------------------------------------------------


def test_no_ink_rows_removed(clean_samples) -> None:
    """Without compression only blank rows are added or removed; ink is copied verbatim."""
    # pad (smaller glyphs), slight trim, and maximal trim (every blank row removed)
    for glyph_range in ((0.45, 0.6), (0.8, 0.8), (1.0, 1.0)):
        step = _step(glyph_only(p_hcompress=0.0, glyph_height_range=glyph_range, min_char_width=1.0))
        for _, text, image in clean_samples:
            inner = _inner(image)
            lo, bg = np.percentile(inner, (1, 90))
            dark = (inner < bg - 0.1 * (bg - lo)).any(axis=1)
            top, bottom = np.flatnonzero(dark)[[0, -1]]
            block = inner[top: bottom + 1]
            out = step(inner, np.random.default_rng(3), len(text))
            assert out.shape[1] == inner.shape[1]
            matches = [k for k in range(out.shape[0] - block.shape[0] + 1)
                       if np.array_equal(out[k: k + block.shape[0]], block)]
            assert matches, "ink rows were altered or clipped"
            k = matches[0]
            rest = np.concatenate([out[:k], out[k + block.shape[0]:]])
            assert (rest >= bg - 0.1 * (bg - lo)).all()  # everything else is blank


def test_compression_keeps_all_ink(clean_samples) -> None:
    step = _step(glyph_only(p_glyph_scale=0.0, hcompress_range=(0.5, 0.7), min_char_width=1.0))
    for _, text, image in clean_samples:
        inner = _inner(image)
        out = step(inner, np.random.default_rng(5), len(text))
        scale = out.shape[1] / inner.shape[1]
        # Total ink scales with the width; clipping a character would lose a chunk of it.
        assert (255.0 - out).sum() == pytest.approx((255.0 - inner).sum() * scale, rel=0.03)
        # Text still starts and ends inside the crop with the original relative margins.
        cols_in = np.flatnonzero((inner < 128).any(axis=0))
        cols_out = np.flatnonzero((out < 128).any(axis=0))
        assert abs(cols_out[0] - cols_in[0] * scale) <= 2
        assert abs((out.shape[1] - cols_out[-1]) - (inner.shape[1] - cols_in[-1]) * scale) <= 2


def test_input_not_modified(clean_samples) -> None:
    field, text, image = next(s for s in clean_samples if s[0] == "aadhaar_number")
    before = np.asarray(image).copy()
    augment_line_image(image, glyph_only(), 0, field_name=field, text=text)
    assert np.array_equal(np.asarray(image), before)


# --- targeting & determinism ---------------------------------------------------


def test_untargeted_fields_identical_to_baseline(clean_samples) -> None:
    exp = dataclasses.replace(BASELINE, p_glyph_scale=1.0, p_hcompress=1.0)
    changed = set()
    for field, text, image in clean_samples:
        for seed in range(4):
            a = np.asarray(augment_line_image(image, BASELINE, [42, 1, seed]))
            b = np.asarray(augment_line_image(image, exp, [42, 1, seed], field_name=field, text=text))
            same = a.shape == b.shape and np.array_equal(a, b)
            if field not in TARGETED:
                assert same, field
            elif not same:
                changed.add(field)
    assert changed == set(TARGETED)


def test_scale_fields_is_respected(clean_samples) -> None:
    cfg = glyph_only(scale_fields=("date_of_birth",))
    for field, text, image in clean_samples:
        out = augment_line_image(image, cfg, 0, field_name=field, text=text)
        plain = augment_line_image(image, GENERIC_OFF, 0)
        assert np.array_equal(np.asarray(out), np.asarray(plain)) == (field != "date_of_birth")


def test_requires_reliable_field_identity(clean_samples) -> None:
    _, text, image = clean_samples[0]
    augmenter = OcrLineAugmenter(glyph_only())
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="field"):
        augmenter(image, rng)  # no metadata -> refuse, never guess
    with pytest.raises(ValueError, match="field"):
        augmenter(image, rng, field_name="aadhaar", text=text)  # not a dataset field
    with pytest.raises(ValueError, match="field"):
        augmenter(image, rng, field_name="name", text="")
    # The baseline augmenter keeps working without metadata.
    OcrLineAugmenter(BASELINE)(image, rng)


def test_deterministic(clean_samples) -> None:
    cfg = dataclasses.replace(BASELINE, p_glyph_scale=0.7, p_hcompress=0.7)
    for field, text, image in clean_samples:
        a = augment_line_image(image, cfg, [42, 3, 7], field_name=field, text=text)
        b = augment_line_image(image, cfg, [42, 3, 7], field_name=field, text=text)
        assert np.array_equal(np.asarray(a), np.asarray(b))
    field, text, image = next(s for s in clean_samples if s[0] == "aadhaar_number")
    outs = {np.asarray(augment_line_image(image, glyph_only(), s, field_name=field, text=text)).tobytes()
            for s in range(6)}
    assert len(outs) > 1


# --- training integration ------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory) -> Path:
    pytest.importorskip("torch")
    from idocr.data.recognition.dataset import RecognitionDatasetConfig, generate_recognition_dataset

    try:
        font = str(resolve_font_path())
    except FileNotFoundError:
        pytest.skip("no OCR font installed")
    config = RecognitionDatasetConfig.from_yaml(ROOT / "configs" / "recognition" / "recognition_dataset.yaml")
    config = dataclasses.replace(config, seed=7, total_samples=120,
                                 generator=dataclasses.replace(config.generator, font_path=font))
    out = tmp_path_factory.mktemp("recognition_glyph")
    generate_recognition_dataset(config, out)
    return out


def _experiment_cfg(dataset_dir: Path, output_dir: Path, **overrides):
    from idocr.training.recognition import load_recognizer_config

    cfg = load_recognizer_config(ROOT / "configs" / "recognition" / "crnn_glyph_scale_spacing_t4.yaml")
    return dataclasses.replace(cfg, dataset_dir=dataset_dir, output_dir=output_dir, device="cpu",
                               amp=False, num_workers=0, **overrides)


def test_only_train_split_is_augmented(dataset_dir: Path, tmp_path: Path) -> None:
    import torch

    from idocr.models.recognizer import Vocabulary
    from idocr.training.recognition.data import RecognitionDataset
    from idocr.training.recognition.train import build_loaders, read_preprocessing

    cfg = _experiment_cfg(dataset_dir, tmp_path)
    train_loader, valid_loader = build_loaders(cfg, dataset_dir, Vocabulary(), HEIGHT,
                                               torch.Generator().manual_seed(42),
                                               preprocessing=read_preprocessing(dataset_dir))
    train_ds, valid_ds = train_loader.dataset, valid_loader.dataset
    assert train_ds.augment is not None and train_ds.augment.glyph_step is not None
    assert valid_ds.augment is None

    # Validation and test items are the stored, preprocessed files, pixel for pixel.
    test_ds = RecognitionDataset(dataset_dir / "test", Vocabulary(), image_height=HEIGHT)
    for ds in (valid_ds, test_ds):
        for i in range(len(ds)):
            image, _, sample = ds[i]
            with Image.open(ds.split_dir / sample.image_path) as stored:
                assert np.array_equal(np.asarray(image), np.asarray(stored.convert("L")))

    # Training: targeted fields change, labels never do.
    train_ds.set_epoch(1)
    vocab = Vocabulary()
    changed = set()
    for i in range(len(train_ds)):
        image, target, sample = train_ds[i]
        assert target == vocab.encode(sample.text)
        assert vocab.decode(target) == sample.text
        assert image.height == HEIGHT
        assert image.width // 4 >= 2 * len(sample.text) + 1
        with Image.open(train_ds.split_dir / sample.image_path) as stored:
            if np.asarray(image).shape != np.asarray(stored).shape:
                changed.add(sample.field_name)
    assert changed  # geometry actually varies on the train split


def test_labels_match_annotations_after_augmentation(dataset_dir: Path) -> None:
    from idocr.models.recognizer import Vocabulary
    from idocr.training.recognition.data import RecognitionDataset, collate_batch

    annotations = [json.loads(line) for line in
                   (dataset_dir / "train" / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    ds = RecognitionDataset(dataset_dir / "train", Vocabulary(), image_height=HEIGHT,
                            augment=OcrLineAugmenter(glyph_only()), seed=42)
    batch = collate_batch([ds[i] for i in range(len(ds))])
    assert batch["texts"] == [a["text"] for a in annotations]
    assert batch["fields"] == [a["field_name"] for a in annotations]


def test_experiment_trains_on_cpu(dataset_dir: Path, tmp_path: Path) -> None:
    from idocr.training.recognition import train

    cfg = _experiment_cfg(dataset_dir, tmp_path, experiment_name="glyph_tiny", epochs=1, batch_size=12,
                          hidden_size=32, lstm_layers=1, max_train_samples=24, max_valid_samples=6)
    record = train(cfg, root=ROOT)
    assert record["status"] == "completed"
    assert record["config"]["augmentation"]["p_glyph_scale"] == 0.7
