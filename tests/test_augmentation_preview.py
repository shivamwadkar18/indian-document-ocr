"""Glyph scale / spacing preview (idocr.data.augmentation.preview + preview script)."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from idocr.data.augmentation.ocr import OcrAugmentConfig
from idocr.data.augmentation.preview import (
    GLYPH_KEYS,
    _font,
    build_variants,
    crop_stats,
    next_preview_dir,
    select_samples,
    wrap_lines,
    write_preview,
)
from idocr.data.preprocessing.ocr import (
    DEFAULT_HORIZONTAL_PADDING,
    DEFAULT_TARGET_HEIGHT,
    DEFAULT_VERTICAL_PADDING,
)
from idocr.data.recognition.generator import GeneratorConfig, SyntheticTextGenerator, resolve_font_path
from idocr.data.recognition.schema import RecognitionSample
from idocr.utils.paths import find_project_root

ROOT = find_project_root()
PREPROCESSING = dict(target_height=DEFAULT_TARGET_HEIGHT, horizontal_padding=DEFAULT_HORIZONTAL_PADDING,
                     vertical_padding=DEFAULT_VERTICAL_PADDING)
PADS = dict(horizontal_padding=DEFAULT_HORIZONTAL_PADDING, vertical_padding=DEFAULT_VERTICAL_PADDING)
FIELDS = ["aadhaar_number", "date_of_birth", "name", "pan_number", "gender", "fathers_name"]
TARGETED = ("aadhaar_number", "date_of_birth", "name")
BASELINE = OcrAugmentConfig(enabled=True)
EXPERIMENT = dataclasses.replace(BASELINE, p_glyph_scale=0.7, p_hcompress=0.7)


def _sample(field: str, i: int, text: str = "X") -> RecognitionSample:
    return RecognitionSample(image_path=f"images/{i:06d}_{field}.png", text=text, field_name=field,
                             document_type="pan", source="synthetic", split="train")


# --- variants / selection / directories ------------------------------------------


def test_build_variants() -> None:
    variants = build_variants(BASELINE, EXPERIMENT, [1, 2])
    assert [v.name for v in variants] == ["original", "baseline", "baseline", "smaller_text", "compressed",
                                          "combined", "experiment", "experiment"]
    by_name = {v.name: v for v in variants}
    assert by_name["original"].config is None
    assert [v.epoch for v in variants if v.name == "experiment"] == [1, 2]
    assert all(v.config is BASELINE for v in variants if v.name == "baseline")
    assert all(v.config is EXPERIMENT for v in variants if v.name == "experiment")
    for name, glyph, compress in (("smaller_text", 1.0, 0.0), ("compressed", 0.0, 1.0), ("combined", 1.0, 1.0)):
        cfg = by_name[name].config
        assert (cfg.p_glyph_scale, cfg.p_hcompress) == (glyph, compress)
        if glyph:  # the range of a step that is on is fixed to its extreme
            assert cfg.glyph_height_range == (0.45, 0.45)
        if compress:
            assert cfg.hcompress_range == (0.5, 0.5)
        generic = [f.name for f in dataclasses.fields(cfg) if f.name.startswith("p_") and f.name not in GLYPH_KEYS]
        assert all(getattr(cfg, f) == 0.0 for f in generic), name
        assert cfg.min_char_width == EXPERIMENT.min_char_width and cfg.scale_fields == EXPERIMENT.scale_fields
    with pytest.raises(ValueError, match="glyph"):
        build_variants(BASELINE, BASELINE, [1])


def test_select_samples_returns_train_indices() -> None:
    samples = [_sample(f, i) for i in range(3) for f in ("name", "gender", "pan_number")]
    assert [(i, s.field_name) for i, s in select_samples(samples, ["pan_number", "name"])] == \
        [(2, "pan_number"), (0, "name")]
    assert [i for i, _ in select_samples(samples, ["name", "gender"], record=2)] == [6, 7]
    with pytest.raises(ValueError, match="aadhaar_number"):
        select_samples(samples, ["name", "aadhaar_number"])
    with pytest.raises(ValueError):
        select_samples(samples, ["name"], record=3)


def test_next_preview_dir_never_reuses(tmp_path: Path) -> None:
    first = next_preview_dir(tmp_path / "exp")
    (first / "keep.png").write_bytes(b"x")
    (tmp_path / "exp" / "preview_notes").mkdir()
    second = next_preview_dir(tmp_path / "exp")
    assert (first.name, second.name) == ("preview_001", "preview_002")
    assert (first / "keep.png").read_bytes() == b"x"
    assert not any(second.iterdir())


def test_wrap_lines_keeps_every_character() -> None:
    font = _font()
    line = "experiment config, epoch 1 (p_glyph=0.7 [0.45, 0.8], p_hcompress=0.7 [0.5, 1.0]) label: 'A B C'"
    wrapped = wrap_lines([line], font, 200)
    assert len(wrapped) > 1
    assert all(font.getlength(w) <= 200 for w in wrapped)
    assert "".join(w.replace(" ", "") for w in wrapped) == line.replace(" ", "")
    assert wrap_lines(["short"], font, 200) == ["short"]


# --- rendering ------------------------------------------------------------------


@pytest.fixture(scope="module")
def split(tmp_path_factory) -> tuple[Path, list[RecognitionSample]]:
    """A tiny train split: one generated record per field, in FIELDS order."""
    try:
        font = str(resolve_font_path())
    except FileNotFoundError:
        pytest.skip("no OCR font installed")
    gen = SyntheticTextGenerator(GeneratorConfig(font_path=font), seed=21)
    root = tmp_path_factory.mktemp("preview_split")
    (root / "images").mkdir()
    samples = []
    for i, field in enumerate(FIELDS):
        s = gen.generate(field)
        sample = _sample(field, i, s.text)
        s.image.save(root / sample.image_path)
        samples.append(sample)
    return root, samples


def _load(root: Path):
    def load(sample):
        with Image.open(root / sample.image_path) as im:
            return im.convert("L")
    return load


def _write(out_dir: Path, split, epochs=(1, 2)):
    root, samples = split
    out_dir.mkdir(parents=True)
    return write_preview(out_dir, list(enumerate(samples)), _load(root),
                         build_variants(BASELINE, EXPERIMENT, list(epochs)), seed=42,
                         preprocessing=PREPROCESSING, scale_fields=TARGETED, meta={"experiment_name": "t"})


def test_write_preview_outputs_and_labels(tmp_path: Path, split) -> None:
    root, samples = split
    out = tmp_path / "p"
    manifest = _write(out, split)
    entries = manifest["entries"]
    assert len(entries) == len(samples) * 8
    assert {p.name for p in out.glob("sheet_*.png")} == {f"sheet_{f}.png" for f in FIELDS}
    assert (out / "preview_sheet.png").is_file()
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest

    text_by_field = {s.field_name: s.text for s in samples}
    for e in entries:
        assert e["label"] == text_by_field[e["field_name"]]  # exact transcription, never altered
        assert e["role"] == ("targeted" if e["field_name"] in TARGETED else "untargeted control")
        image = Image.open(out / e["file"])
        assert image.size == tuple(e["size"]) and image.height == 56
        assert e["ctc_frames"] >= e["ctc_frames_needed"]
        if e["variant"] == "original":
            assert e["seed"] is None and e["unchanged_from_original"]
            assert np.array_equal(np.asarray(image), np.asarray(_load(root)(samples[e["train_index"]])))
        else:  # training convention [seed, epoch, index]; isolated variants use epoch 0
            seed, epoch, index = e["seed"]
            assert (seed, index) == (42, e["train_index"])
            assert epoch in ((1, 2) if e["variant"] in ("baseline", "experiment") else (0,))


def test_controls_unchanged_and_targets_transformed(tmp_path: Path, split) -> None:
    entries = _write(tmp_path / "p", split)["entries"]
    for e in entries:
        isolated = e["variant"] in ("smaller_text", "compressed", "combined")
        if e["field_name"] not in TARGETED:
            if isolated:
                assert e["unchanged_from_original"], e
            if e["variant"] == "experiment":
                assert e["identical_to_baseline"] is True
        elif isolated:
            assert not e["unchanged_from_original"], e
            if e["variant"] != "compressed":
                assert e["ink_height_fraction"] < 0.5  # 0.45 configured; measured after resampling
            if e["variant"] != "smaller_text":
                assert e["px_per_char"] >= EXPERIMENT.min_char_width - 0.5
    original = {e["field_name"]: e for e in entries if e["variant"] == "original"}
    for e in entries:
        if e["field_name"] in TARGETED and e["variant"] == "compressed":
            assert e["size"][0] < original[e["field_name"]]["size"][0]


def test_write_preview_deterministic_and_refuses_non_empty(tmp_path: Path, split) -> None:
    a = _write(tmp_path / "a", split)
    b = _write(tmp_path / "b", split)
    for ea, eb in zip(a["entries"], b["entries"]):
        assert (tmp_path / "a" / ea["file"]).read_bytes() == (tmp_path / "b" / eb["file"]).read_bytes()
    root, samples = split
    with pytest.raises(FileExistsError):
        write_preview(tmp_path / "a", list(enumerate(samples)), _load(root), build_variants(BASELINE, EXPERIMENT, [1]),
                      seed=42, preprocessing=PREPROCESSING, scale_fields=TARGETED, meta={})


def test_crop_stats(split) -> None:
    root, samples = split
    image = _load(root)(samples[0])
    stats = crop_stats(image, samples[0].text, **PADS)
    assert 0.65 <= stats["ink_height_fraction"] <= 0.8  # synthetic glyphs ~75% of input height
    assert stats["px_per_char"] == round((image.width - 16) / len(samples[0].text), 1)
    assert stats["ctc_frames"] == image.width // 4
    blank = Image.new("L", (100, 56), 255)
    assert crop_stats(blank, "AB", **PADS)["ink_height_fraction"] is None


# --- script --------------------------------------------------------------------


def test_script_writes_new_preview_dir_each_run(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from idocr.data.recognition.dataset import RecognitionDatasetConfig, generate_recognition_dataset

    try:
        font = str(resolve_font_path())
    except FileNotFoundError:
        pytest.skip("no OCR font installed")
    config = RecognitionDatasetConfig.from_yaml(ROOT / "configs" / "recognition" / "recognition_dataset.yaml")
    config = dataclasses.replace(config, seed=3, total_samples=60,
                                 generator=dataclasses.replace(config.generator, font_path=font))
    dataset = tmp_path / "recognition"
    generate_recognition_dataset(config, dataset)
    before = sorted(p.name for p in dataset.rglob("*"))

    cmd = [sys.executable, str(ROOT / "scripts" / "preview_ocr_augmentation.py"),
           "--config", str(ROOT / "configs" / "recognition" / "crnn_glyph_scale_spacing_t4.yaml"),
           "--dataset-dir", str(dataset), "--out-root", str(tmp_path / "previews"), "--epochs", "1"]
    for expected in ("preview_001", "preview_002"):
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        out = tmp_path / "previews" / expected
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["experiment_name"] == "crnn_glyph_scale_spacing_t4_10ep"
        assert [e["field_name"] for e in manifest["entries"][::6]] == FIELDS
        assert {e["train_index"] for e in manifest["entries"]} == set(range(6))
        assert manifest["seed"] == 42
    assert sorted(p.name for p in dataset.rglob("*")) == before  # dataset only read
