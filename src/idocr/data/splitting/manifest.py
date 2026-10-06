"""Manifest generation for source groups and split assignments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from idocr.data.splitting.grouping import SourceGroup


MANIFEST_FIELDNAMES = [
    "dataset",
    "image_rel_path",
    "filename",
    "raw_split",
    "target_split",
    "source_group_id",
    "grouping_method",
    "confidence",
    "sha256",
    "dhash",
    "width",
    "height",
    "annotation_count",
    "notes",
]


def create_manifest_records(
    splits: Mapping[str, Sequence[SourceGroup]],
    document_type: str,
) -> list[dict]:
    """Generate tabular records mapping each image to its group and target split."""
    rows: list[dict] = []
    for split_name, groups in splits.items():
        for group in groups:
            for member in group.members:
                rows.append({
                    "dataset": document_type,
                    "image_rel_path": member.rel_path,
                    "filename": member.path.name,
                    "raw_split": member.raw_split,
                    "target_split": split_name,
                    "source_group_id": group.group_id,
                    "grouping_method": group.grouping_method,
                    "confidence": group.confidence,
                    "sha256": member.sha256,
                    "dhash": f"{member.dhash:016x}" if member.dhash is not None else "",
                    "width": member.width,
                    "height": member.height,
                    "annotation_count": member.annotation_count,
                    "notes": group.notes,
                })
    # Sort by source_group_id and filename for clean presentation
    return sorted(rows, key=lambda r: (r["source_group_id"], r["filename"]))


def write_manifest_csv(
    output_path: str | Path,
    splits: Mapping[str, Sequence[SourceGroup]],
    document_type: str,
) -> Path:
    """Write split manifest to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = create_manifest_records(splits, document_type)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    return output_path


def load_manifest_csv(manifest_path: str | Path) -> list[dict]:
    """Load records from a manifest CSV."""
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
