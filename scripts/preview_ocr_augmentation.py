"""Write a small clean-vs-augmented preview of OCR training augmentation.

Usage:
    .venv\Scripts\python scripts\preview_ocr_augmentation.py [--records 2] [--seed 0]
    python scripts/preview_ocr_augmentation.py --config configs/recognition/crnn_glyph_scale_spacing_t4.yaml
        [--record 0] [--epochs 1 2 3]

Without --config: takes the first N records (6 fields each) of the existing
train split, applies the default OcrAugmentConfig (enabled) and writes, under
data/processed/recognition/augment_preview/ (replaced each run):
    <index>_<field>_clean.png, <index>_<field>_aug.png, preview_sheet.png

With --config (an experiment using the glyph scale / spacing step): for one
train sample per field (targeted fields first, then untargeted controls) writes
original / baseline / smaller_text / compressed / combined / experiment
variants, labelled with the transcription, settings, seed and measurements, to
a new experiments/previews/<experiment_name>/preview_NNN/ (never overwritten):
    <index>_<field>_<k>_<variant>.png, sheet_<field>.png, preview_sheet.png,
    manifest.json
Baseline and experiment rows use the training seeds [seed, epoch, index].
See idocr.data.augmentation.preview. The dataset itself is only read.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from idocr.data.augmentation.ocr import OcrAugmentConfig, OcrLineAugmenter
from idocr.data.augmentation.preview import build_variants, next_preview_dir, select_samples, write_preview
from idocr.data.recognition.dataset import INFO_FILENAME
from idocr.training.recognition import load_recognizer_config
from idocr.training.recognition.data import read_split
from idocr.training.recognition.train import read_preprocessing
from idocr.utils.paths import find_project_root

GAP = 6
DEFAULT_BASELINE_CONFIG = Path("configs/recognition/crnn_mixedcase_augmented_t4.yaml")
CONTROL_FIELDS = ("pan_number", "gender", "fathers_name")


def experiment_preview(args: argparse.Namespace) -> int:
    root = find_project_root()
    cfg = load_recognizer_config(args.config)
    baseline_cfg = load_recognizer_config(args.baseline_config)
    experiment, baseline = cfg.augment_config, baseline_cfg.augment_config
    dataset_dir = args.dataset_dir or (cfg.dataset_dir if cfg.dataset_dir.is_absolute() else root / cfg.dataset_dir)
    if not (dataset_dir / INFO_FILENAME).is_file():
        print(f"ERROR: no recognition dataset at {dataset_dir}")
        return 1
    seed = cfg.seed if args.seed is None else args.seed
    fields = list(experiment.scale_fields) + [f for f in CONTROL_FIELDS if f not in experiment.scale_fields]
    preprocessing = read_preprocessing(dataset_dir)
    split_dir = dataset_dir / "train"
    # Whole split, so the indices (and seeds) are the ones training uses.
    selected = select_samples(read_split(split_dir), fields, args.record)
    variants = build_variants(baseline, experiment, args.epochs)

    def load_image(sample):
        with Image.open(split_dir / sample.image_path) as im:
            return im.convert("L")

    out_dir = next_preview_dir(args.out_root or root / "experiments" / "previews" / cfg.experiment_name)
    manifest = write_preview(
        out_dir, selected, load_image, variants, seed=seed, preprocessing=preprocessing,
        scale_fields=experiment.scale_fields,
        meta={"experiment_name": cfg.experiment_name, "config": Path(args.config).as_posix(),
              "baseline_config": Path(args.baseline_config).as_posix(),
              "dataset_dir": Path(dataset_dir).as_posix(), "split": "train", "record": args.record},
    )
    for e in manifest["entries"]:
        print(f"{e['field_name']:15s} {e['variant']:13s} seed={str(e['seed']):14s} "
              f"{e['size'][0]:4d}x{e['size'][1]} ink={e['ink_height_fraction']} "
              f"{e['px_per_char']:5.1f}px/char CTC {e['ctc_frames']}/{e['ctc_frames_needed']}"
              + ("  unchanged" if e["seed"] and e["unchanged_from_original"] else ""))
    print(f"\nWrote {len(manifest['entries'])} crops, {len(selected)} field sheets, "
          f"preview_sheet.png and manifest.json to {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="default: data/processed/recognition, or the config's dataset_dir")
    parser.add_argument("--records", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None, help="default: 0, or the config's seed")
    parser.add_argument("--config", type=Path, help="experiment recognizer config (glyph scale / spacing)")
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--record", type=int, default=0, help="--config: which sample of each field")
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3],
                        help="--config: training epochs whose augmentation draws are shown")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="--config: default experiments/previews/<experiment_name>")
    args = parser.parse_args()
    if args.config is not None:
        return experiment_preview(args)
    args.dataset_dir = args.dataset_dir or Path("data/processed/recognition")
    args.seed = 0 if args.seed is None else args.seed

    if not (args.dataset_dir / INFO_FILENAME).is_file():
        print(f"ERROR: no recognition dataset at {args.dataset_dir}")
        return 1
    out_dir = args.dataset_dir / "augment_preview"
    if out_dir.exists():
        shutil.rmtree(out_dir)  # preview only; regenerated each run
    out_dir.mkdir()

    preprocessing = read_preprocessing(args.dataset_dir)
    config = dataclasses.replace(OcrAugmentConfig(), enabled=True)
    augmenter = OcrLineAugmenter(config, **preprocessing)
    split_dir = args.dataset_dir / "train"
    samples = read_split(split_dir, args.records * 6)

    rows = []
    for index, sample in enumerate(samples):
        with Image.open(split_dir / sample.image_path) as im:
            clean = im.convert("L")
        augmented = augmenter(clean, np.random.default_rng([args.seed, 0, index]))
        stem = f"{index:02d}_{sample.field_name}"
        clean.save(out_dir / f"{stem}_clean.png")
        augmented.save(out_dir / f"{stem}_aug.png")
        rows.append((clean, augmented))
        print(f"{stem:22s} {sample.text!r:20s} clean {clean.size[0]}x{clean.size[1]}  "
              f"aug {augmented.size[0]}x{augmented.size[1]} {augmented.mode}")

    left = max(c.width for c, _ in rows)
    width = left + GAP + max(a.width for _, a in rows)
    sheet = Image.new("L", (width, sum(c.height + GAP for c, _ in rows)), 128)
    y = 0
    for clean, augmented in rows:
        sheet.paste(clean, (0, y))
        sheet.paste(augmented, (left + GAP, y))
        y += clean.height + GAP
    sheet.save(out_dir / "preview_sheet.png")
    print(f"\nWrote {2 * len(rows)} images + preview_sheet.png to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
