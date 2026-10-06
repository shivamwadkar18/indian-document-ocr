"""Builds processed dataset directories from source group splits."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping, Sequence

from idocr.data.splitting.grouping import SourceGroup


def build_processed_dataset(
    splits: Mapping[str, Sequence[SourceGroup]],
    output_dir: str | Path,
    overwrite: bool = True,
) -> dict:
    """Copy images and exact labels to destination processed split folders.

    Preserves label content byte-for-byte, including class IDs, YOLO coordinates,
    polygons, and unassigned classes.

    Args:
        splits: Mapping from split name ('train', 'valid', 'test') to SourceGroups.
        output_dir: Directory where processed dataset will be placed.
        overwrite: If True, existing output split directories will be cleaned first.

    Returns:
        Summary dictionary with counts of copied images and labels per split.
    """
    output_dir = Path(output_dir)
    summary: dict[str, dict[str, int]] = {}

    for split_name, groups in splits.items():
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"

        if overwrite and (output_dir / split_name).exists():
            shutil.rmtree(output_dir / split_name)

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        copied_images = 0
        copied_labels = 0

        for group in groups:
            for member in group.members:
                dst_img = img_dir / member.path.name
                shutil.copy2(member.path, dst_img)
                copied_images += 1

                if member.label_path and member.label_path.exists():
                    dst_lbl = lbl_dir / member.label_path.name
                    shutil.copy2(member.label_path, dst_lbl)
                    copied_labels += 1

        summary[split_name] = {
            "images": copied_images,
            "labels": copied_labels,
            "groups": len(groups),
        }

    return summary
