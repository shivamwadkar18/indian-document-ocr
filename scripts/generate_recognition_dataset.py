"""Generate and validate the synthetic field-level OCR recognition dataset.

Usage:
    python scripts/generate_recognition_dataset.py
    python scripts/generate_recognition_dataset.py --overwrite
    python scripts/generate_recognition_dataset.py --seed 7 --total-samples 600

Sample count, seed, output directory and font come from
configs/recognition/recognition_dataset.yaml; CLI flags override them.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

from idocr.data.recognition.dataset import (
    SPLITS,
    RecognitionDatasetConfig,
    generate_recognition_dataset,
    validate_recognition_dataset,
)
from idocr.data.recognition.generator import SUPPORTED_FIELDS


def _snapshot_tree_stats(root: Path) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) for every file under root."""
    if not root.exists():
        return {}
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def _is_within(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/recognition/recognition_dataset.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-samples", type=int, default=None)
    parser.add_argument("--font-path", type=str, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated dataset (train/, valid/, test/, "
        "dataset_info.json only).",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    config = RecognitionDatasetConfig.from_yaml(args.config)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.total_samples is not None:
        overrides["total_samples"] = args.total_samples
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.font_path is not None:
        overrides["generator"] = dataclasses.replace(
            config.generator, font_path=args.font_path
        )
    config = dataclasses.replace(config, **overrides)

    if _is_within(config.output_dir, args.raw_dir):
        print(f"[FAILED] output dir {config.output_dir} is inside {args.raw_dir}")
        return 1

    print("============ GENERATING OCR RECOGNITION DATASET ============", flush=True)
    print(f"Config:     {args.config}")
    print(f"Output dir: {config.output_dir}")
    print(f"Seed:       {config.seed}")
    print(f"Samples:    {config.total_samples}")

    raw_before = _snapshot_tree_stats(args.raw_dir)
    started = time.perf_counter()
    try:
        info = generate_recognition_dataset(
            config, overwrite=args.overwrite, progress=True
        )
    except FileExistsError as exc:
        print(f"\n[FAILED] {exc}")
        return 1
    elapsed = time.perf_counter() - started

    print("\nValidating...", flush=True)
    result = validate_recognition_dataset(config)
    if _snapshot_tree_stats(args.raw_dir) != raw_before:
        result.errors.append("raw data changed during generation")

    gen = info["generator_config"]
    print("\n======================= REPORT =======================")
    print(f"Seed:              {info['seed']}")
    print(f"Font:              {info['font_file']}")
    print(
        f"Generator:         idocr.data.recognition.generator.SyntheticTextGenerator "
        f"(font_size={gen['font_size']}, rotation<=+/-{gen['max_rotation_degrees']}deg, "
        f"blur<={gen['max_blur_radius']}, noise_std<={gen['max_noise_std']}, "
        f"bg={tuple(gen['background_range'])}, ink={tuple(gen['ink_range'])})"
    )
    print(
        f"Preprocessing:     target_height={gen['target_height']}, "
        f"horizontal_padding={gen['horizontal_padding']}, "
        f"vertical_padding={gen['vertical_padding']}"
    )
    print(f"Generation time:   {elapsed:.0f}s")
    print(f"Total images:      {result.total_images}")
    print(f"Total annotations: {result.total_annotations}")
    for split in SPLITS:
        print(
            f"  {split:5s}  images={result.images_by_split.get(split, 0):6d}  "
            f"annotations={result.annotations_by_split.get(split, 0):6d}"
        )

    header = "field".ljust(16) + "".join(s.rjust(8) for s in (*SPLITS, "total"))
    print("\nPer-field counts:")
    print("  " + header)
    for field_name in SUPPORTED_FIELDS:
        counts = [result.by_split_and_field.get(s, {}).get(field_name, 0) for s in SPLITS]
        row = field_name.ljust(16) + "".join(str(c).rjust(8) for c in (*counts, sum(counts)))
        print("  " + row)

    print(f"\nDocument types:    {result.by_document_type}")
    print(f"Image modes:       {sorted(result.modes)}")
    print(f"Width range:       {result.width_range}")
    print(f"Height range:      {result.height_range}")

    if not result.is_valid:
        print(f"\n[FAILED] {len(result.errors)} validation errors:")
        for err in result.errors[:50]:
            print(f"  - {err}")
        return 1

    print("\n[SUCCESS] Dataset generated and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
