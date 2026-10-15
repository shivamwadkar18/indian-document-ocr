"""Before/after previews of OCR augmentation for one experiment config.

Used by ``scripts/preview_ocr_augmentation.py --config ...``. For one
synthetic train sample per field, each row is one *variant*:

- ``original``      the stored crop, no augmentation;
- ``baseline``      the baseline config's train-time augmentation;
- ``smaller_text``  only the glyph-scale step, at the smallest configured height;
- ``compressed``    only the horizontal compression, at the strongest setting;
- ``combined``      both, at those extremes (``min_char_width`` still applies);
- ``experiment``    the experiment config's full train-time augmentation.

``baseline`` and ``experiment`` use the training seed convention
``[seed, epoch, index]`` (``index`` = position in the train split), so a
row shows exactly what that sample gets in that epoch. The isolated
variants fix their ranges to a single value, so they do not depend on the seed.

Previews go to a new ``preview_NNN`` directory; existing ones are never touched.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from idocr.data.augmentation.ocr import OcrAugmentConfig, OcrLineAugmenter
from idocr.data.recognition.schema import RecognitionSample

GLYPH_KEYS = ("p_glyph_scale", "p_hcompress")
PREVIEW_DIR_PREFIX = "preview_"
TEXT_COLUMN_WIDTH = 430
LINE_HEIGHT = 15
GAP = 6


@dataclass(frozen=True)
class Variant:
    name: str
    settings: str  # human-readable transformation settings
    config: OcrAugmentConfig | None  # None = the stored crop
    epoch: int = 0


def generic_off(config: OcrAugmentConfig) -> OcrAugmentConfig:
    """``config`` with every generic step disabled (glyph keys untouched)."""
    return dataclasses.replace(config, **{
        f.name: 0.0 for f in dataclasses.fields(OcrAugmentConfig)
        if f.name.startswith("p_") and f.name not in GLYPH_KEYS
    })


def build_variants(baseline: OcrAugmentConfig, experiment: OcrAugmentConfig,
                   epochs: Sequence[int]) -> list[Variant]:
    if not experiment.glyph_scale_active:
        raise ValueError("experiment config has no glyph scale / spacing step enabled")
    frac = experiment.glyph_height_range[0]
    scale = experiment.hcompress_range[0]
    isolated = generic_off(experiment)
    floor = f"min_char_width={experiment.min_char_width:g}px"
    variants = [Variant("original", "stored crop, no augmentation", None)]
    variants += [Variant("baseline", f"baseline config, generic steps, epoch {e}", baseline, e) for e in epochs]
    variants += [
        Variant("smaller_text", f"glyph height={frac:.2f} of input, generic off, {floor}",
                dataclasses.replace(isolated, p_glyph_scale=1.0, p_hcompress=0.0,
                                    glyph_height_range=(frac, frac))),
        Variant("compressed", f"hcompress={scale:.2f}, generic off, {floor}",
                dataclasses.replace(isolated, p_glyph_scale=0.0, p_hcompress=1.0,
                                    hcompress_range=(scale, scale))),
        Variant("combined", f"glyph height={frac:.2f} + hcompress={scale:.2f}, generic off, {floor}",
                dataclasses.replace(isolated, p_glyph_scale=1.0, p_hcompress=1.0,
                                    glyph_height_range=(frac, frac), hcompress_range=(scale, scale))),
    ]
    variants += [
        Variant("experiment",
                f"experiment config, epoch {e} (p_glyph={experiment.p_glyph_scale:g} "
                f"{list(experiment.glyph_height_range)}, p_hcompress={experiment.p_hcompress:g} "
                f"{list(experiment.hcompress_range)})", experiment, e)
        for e in epochs
    ]
    return variants


def select_samples(samples: Sequence[RecognitionSample], fields: Sequence[str],
                   record: int = 0) -> list[tuple[int, RecognitionSample]]:
    """``(train index, sample)`` of the ``record``-th sample of each field, in ``fields`` order."""
    seen: dict[str, int] = {}
    chosen: dict[str, tuple[int, RecognitionSample]] = {}
    for index, sample in enumerate(samples):
        if sample.field_name in fields and sample.field_name not in chosen:
            if seen.get(sample.field_name, 0) == record:
                chosen[sample.field_name] = (index, sample)
            seen[sample.field_name] = seen.get(sample.field_name, 0) + 1
    missing = [f for f in fields if f not in chosen]
    if missing:
        raise ValueError(f"no sample #{record} for fields {missing}")
    return [chosen[f] for f in fields]


def crop_stats(image: Image.Image, text: str, *, horizontal_padding: int,
               vertical_padding: int) -> dict[str, Any]:
    """Size, ink height / input height, px per character and CTC frame margin."""
    arr = np.asarray(image.convert("L"))
    inner = arr[vertical_padding: arr.shape[0] - vertical_padding,
                horizontal_padding: arr.shape[1] - horizontal_padding]
    lo, bg = np.percentile(inner, (1, 90))
    rows = np.flatnonzero((inner < (lo + bg) / 2).any(axis=1)) if bg - lo >= 30 else np.array([])
    return {
        "size": [image.width, image.height],
        "ink_height_fraction": round(float(rows[-1] - rows[0] + 1) / arr.shape[0], 3) if rows.size else None,
        "px_per_char": round(inner.shape[1] / len(text), 1),
        "ctc_frames": image.width // 4,
        "ctc_frames_needed": 2 * len(text) + 1,
    }


def render_variant(image: Image.Image, sample: RecognitionSample, variant: Variant,
                   index: int, seed: int, preprocessing: dict[str, int]) -> tuple[Image.Image, list[int] | None]:
    """Augmented image and the seed used (None for the stored crop)."""
    if variant.config is None:
        return image, None
    rng_seed = [seed, variant.epoch, index]
    out = OcrLineAugmenter(variant.config, **preprocessing)(
        image, np.random.default_rng(rng_seed), field_name=sample.field_name, text=sample.text)
    return out, rng_seed


def next_preview_dir(root: Path) -> Path:
    """Create and return ``root/preview_NNN`` with the next unused number."""
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name[len(PREVIEW_DIR_PREFIX):]) for p in root.glob(f"{PREVIEW_DIR_PREFIX}*")
               if p.name[len(PREVIEW_DIR_PREFIX):].isdigit()]
    path = root / f"{PREVIEW_DIR_PREFIX}{max(numbers, default=0) + 1:03d}"
    path.mkdir()  # never reuses an existing directory
    return path


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default(size=12)


def wrap_lines(lines: Sequence[str], font: ImageFont.ImageFont, width: int) -> list[str]:
    """Word-wrap each line to ``width`` px; nothing is dropped (long words stay whole)."""
    out: list[str] = []
    for line in lines:
        current = ""
        for word in line.split(" "):
            candidate = f"{current} {word}" if current else word
            if current and font.getlength(candidate) > width:
                out.append(current)
                current = "  " + word  # indent continuation lines
            else:
                current = candidate
        out.append(current)
    return out


def compose_sheet(title: str, rows: Sequence[tuple[list[str], Image.Image]]) -> Image.Image:
    """White sheet: a text column (variant, settings, label, stats) left of each crop."""
    font = _font()
    rows = [(wrap_lines(lines, font, TEXT_COLUMN_WIDTH - 2 * GAP), im) for lines, im in rows]
    header_h = 2 * LINE_HEIGHT
    heights = [max(im.height, len(lines) * LINE_HEIGHT) + GAP for lines, im in rows]
    width = TEXT_COLUMN_WIDTH + max(im.width for _, im in rows) + GAP
    sheet = Image.new("L", (width, header_h + sum(heights)), 255)
    draw = ImageDraw.Draw(sheet)
    draw.text((GAP, GAP // 2), title, fill=0, font=font)
    y = header_h
    for (lines, im), h in zip(rows, heights):
        draw.line([(0, y - GAP // 2), (width, y - GAP // 2)], fill=200)
        for i, line in enumerate(lines):
            draw.text((GAP, y + i * LINE_HEIGHT), line, fill=0, font=font)
        sheet.paste(im, (TEXT_COLUMN_WIDTH, y))
        # Frame the crop so its own padding is visible against the sheet.
        draw.rectangle([TEXT_COLUMN_WIDTH - 1, y - 1, TEXT_COLUMN_WIDTH + im.width, y + im.height], outline=128)
        y += h
    return sheet


def write_preview(
    out_dir: Path,
    selected: Sequence[tuple[int, RecognitionSample]],
    load_image,
    variants: Sequence[Variant],
    *,
    seed: int,
    preprocessing: dict[str, int],
    scale_fields: Sequence[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Render every variant of every selected sample into ``out_dir`` (must exist, empty)."""
    if any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    pads = dict(horizontal_padding=preprocessing["horizontal_padding"],
                vertical_padding=preprocessing["vertical_padding"])
    entries: list[dict[str, Any]] = []
    sheets = []
    for index, sample in selected:
        image = load_image(sample)
        targeted = sample.field_name in scale_fields
        role = "targeted" if targeted else "untargeted control"
        rows = []
        baseline_by_epoch: dict[int, np.ndarray] = {}
        for k, variant in enumerate(variants):
            out, rng_seed = render_variant(image, sample, variant, index, seed, preprocessing)
            arr = np.asarray(out)
            if variant.name == "baseline":
                baseline_by_epoch[variant.epoch] = arr
            stats = crop_stats(out, sample.text, **pads)
            same_as_original = arr.shape == np.asarray(image).shape and np.array_equal(arr, np.asarray(image))
            ref = baseline_by_epoch.get(variant.epoch) if variant.name == "experiment" else None
            same_as_baseline = None if ref is None else bool(arr.shape == ref.shape and np.array_equal(arr, ref))
            file_name = f"{index:06d}_{sample.field_name}_{k:02d}_{variant.name}.png"
            out.save(out_dir / file_name)
            entry = {
                "field_name": sample.field_name, "role": role, "train_index": index,
                "image_path": sample.image_path, "label": sample.text, "variant": variant.name,
                "settings": variant.settings, "seed": rng_seed, "file": file_name,
                "unchanged_from_original": bool(same_as_original),
                "identical_to_baseline": same_as_baseline, **stats,
            }
            entries.append(entry)
            flags = []
            if variant.config is not None and same_as_original:
                flags.append("UNCHANGED" if targeted else "UNCHANGED (field not targeted)")
            if same_as_baseline is not None:
                flags.append("== baseline" if same_as_baseline else "!= baseline")
            ink = stats["ink_height_fraction"]
            rows.append(([
                f"[{variant.name}] seed={rng_seed}" + (f"  {' '.join(flags)}" if flags else ""),
                variant.settings,
                f"label: {sample.text!r} ({len(sample.text)} chars)",
                f"{stats['size'][0]}x{stats['size'][1]}  ink={ink if ink is None else f'{ink:.2f}'}"
                f"  {stats['px_per_char']}px/char  CTC {stats['ctc_frames']}/{stats['ctc_frames_needed']}",
            ], out))
        sheet = compose_sheet(f"{sample.field_name} ({role})  train index {index}  {sample.image_path}", rows)
        sheet.save(out_dir / f"sheet_{sample.field_name}.png")
        sheets.append(sheet)

    overview = Image.new("L", (max(s.width for s in sheets), sum(s.height + GAP for s in sheets)), 160)
    y = 0
    for s in sheets:
        overview.paste(s, (0, y))
        y += s.height + GAP
    overview.save(out_dir / "preview_sheet.png")

    manifest = {**meta, "seed": seed, "preprocessing": preprocessing, "scale_fields": list(scale_fields),
                "variants": [{"name": v.name, "settings": v.settings, "epoch": v.epoch} for v in variants],
                "entries": entries}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
