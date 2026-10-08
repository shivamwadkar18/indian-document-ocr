"""Write a small clean-vs-augmented preview of OCR training augmentation.

Usage:
    .venv\Scripts\python scripts\preview_ocr_augmentation.py [--records 2] [--seed 0]

Takes the first N records (6 fields each) of the existing train split, applies
the default OcrAugmentConfig (enabled) and writes, under
data/processed/recognition/augment_preview/:
    <index>_<field>_clean.png, <index>_<field>_aug.png, preview_sheet.png
The dataset itself is only read.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from idocr.data.augmentation.ocr import OcrAugmentConfig, OcrLineAugmenter
from idocr.data.recognition.dataset import INFO_FILENAME
from idocr.training.recognition.data import read_split
from idocr.training.recognition.train import read_preprocessing

GAP = 6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/recognition"))
    parser.add_argument("--records", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
