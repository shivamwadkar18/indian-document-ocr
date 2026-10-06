"""Source grouping logic for dataset images to prevent train/eval leakage."""

from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

from idocr.data.audit.naming import roboflow_source_stem
from idocr.data.splitting.similarity import difference_hash, sha256_file


ConfidenceLevel = Literal["high", "possible", "unmatched"]
GroupingMethod = Literal["roboflow_source_stem", "exact_byte_duplicate", "near_duplicate", "single_source"]


@dataclass
class ImageRecord:
    """Metadata for a single image in the dataset."""
    path: Path
    rel_path: str
    raw_split: str
    stem: str
    roboflow_source: str | None
    sha256: str
    dhash: int | None
    width: int
    height: int
    label_path: Path | None
    annotation_count: int


@dataclass
class SourceGroup:
    """A collection of images representing variants/copies of the same underlying document."""
    group_id: str
    document_type: str
    members: list[ImageRecord] = field(default_factory=list)
    grouping_method: GroupingMethod = "single_source"
    confidence: ConfidenceLevel = "unmatched"
    notes: str = ""

    @property
    def image_count(self) -> int:
        return len(self.members)

    @property
    def raw_splits(self) -> set[str]:
        return {m.raw_split for m in self.members}

    @property
    def is_cross_split_in_raw(self) -> bool:
        return len(self.raw_splits) > 1


def scan_image_records(root_dir: Path) -> list[ImageRecord]:
    """Scan all images and labels in a raw dataset directory."""
    root_dir = Path(root_dir)
    image_paths = sorted(root_dir.glob("*/**/images/*.*"))
    records: list[ImageRecord] = []

    for img_path in image_paths:
        raw_split = img_path.parent.parent.name
        rel_path = img_path.relative_to(root_dir).as_posix()
        stem = img_path.stem
        rf_source = roboflow_source_stem(stem)

        # Matching label
        lbl_path = img_path.parent.parent / "labels" / f"{stem}.txt"
        ann_count = 0
        if lbl_path.exists():
            lines = [l for l in lbl_path.read_text("utf-8-sig").splitlines() if l.strip()]
            ann_count = len(lines)
        else:
            lbl_path = None

        # Image bytes and hashes
        data = img_path.read_bytes()
        digest = sha256_file(img_path)
        dhash_val = None
        width, height = 0, 0
        try:
            with Image.open(io.BytesIO(data)) as img:
                width, height = img.size
                dhash_val = difference_hash(img)
        except Exception:
            pass

        records.append(
            ImageRecord(
                path=img_path,
                rel_path=rel_path,
                raw_split=raw_split,
                stem=stem,
                roboflow_source=rf_source,
                sha256=digest,
                dhash=dhash_val,
                width=width,
                height=height,
                label_path=lbl_path,
                annotation_count=ann_count,
            )
        )

    return records


def group_dataset(
    root_dir: Path,
    document_type: str,
) -> tuple[list[SourceGroup], list[dict]]:
    """Group images by underlying source document.

    Returns:
        tuple of (source_groups, ambiguous_pairs)
    """
    records = scan_image_records(root_dir)
    groups: list[SourceGroup] = []
    ambiguous_pairs: list[dict] = []

    if document_type.lower() == "aadhaar":
        # In Aadhaar, Roboflow exported multiple augmentations per uploaded original document.
        # Group strictly by Roboflow source stem.
        by_source: dict[str, list[ImageRecord]] = defaultdict(list)
        for r in records:
            key = r.roboflow_source if r.roboflow_source is not None else r.stem
            by_source[key].append(r)

        for idx, (src_key, members) in enumerate(sorted(by_source.items(), key=lambda x: x[0]), start=1):
            group_id = f"aadhaar_src_{idx:04d}"
            method: GroupingMethod = "roboflow_source_stem" if len(members) > 1 else "single_source"
            confidence: ConfidenceLevel = "high" if len(members) > 1 else "unmatched"
            note = f"Augmented copies of source '{src_key}'" if len(members) > 1 else f"Single image source '{src_key}'"
            groups.append(
                SourceGroup(
                    group_id=group_id,
                    document_type="aadhaar",
                    members=members,
                    grouping_method=method,
                    confidence=confidence,
                    notes=note,
                )
            )

    elif document_type.lower() == "pan":
        # In PAN, group byte-identical duplicates (124 pairs).
        # Single images remain unmatched single sources.
        by_source: dict[str, list[ImageRecord]] = defaultdict(list)
        for r in records:
            key = r.roboflow_source if r.roboflow_source is not None else r.stem
            by_source[key].append(r)

        for idx, (src_key, members) in enumerate(sorted(by_source.items(), key=lambda x: x[0]), start=1):
            group_id = f"pan_src_{idx:04d}"
            if len(members) > 1:
                # Check if exact byte duplicates
                sha_set = {m.sha256 for m in members}
                if len(sha_set) == 1:
                    method = "exact_byte_duplicate"
                    confidence = "high"
                    note = f"Exact byte duplicate pair for source '{src_key}'"
                else:
                    method = "roboflow_source_stem"
                    confidence = "high"
                    note = f"Variants for source '{src_key}'"
            else:
                method = "single_source"
                confidence = "unmatched"
                note = f"Single image source '{src_key}'"

            groups.append(
                SourceGroup(
                    group_id=group_id,
                    document_type="pan",
                    members=members,
                    grouping_method=method,
                    confidence=confidence,
                    notes=note,
                )
            )

    else:
        # Generic grouping by stem
        for idx, r in enumerate(records, start=1):
            groups.append(
                SourceGroup(
                    group_id=f"{document_type}_src_{idx:04d}",
                    document_type=document_type,
                    members=[r],
                    grouping_method="single_source",
                    confidence="unmatched",
                    notes="Single image",
                )
            )

    # Check for ambiguous near-duplicates across different source groups
    rep_map = {g.group_id: g.members[0] for g in groups if g.members[0].dhash is not None}
    items = list(rep_map.items())
    for i in range(len(items)):
        g1, r1 = items[i]
        for j in range(i + 1, len(items)):
            g2, r2 = items[j]
            dist = (r1.dhash ^ r2.dhash).bit_count()  # type: ignore[operator]
            if dist <= 2:
                ambiguous_pairs.append({
                    "group_1": g1,
                    "group_2": g2,
                    "file_1": r1.rel_path,
                    "file_2": r2.rel_path,
                    "dhash_distance": dist,
                    "same_size": (r1.width == r2.width and r1.height == r2.height),
                    "size_1": f"{r1.width}x{r1.height}",
                    "size_2": f"{r2.width}x{r2.height}",
                    "note": "Near visual similarity (dHash <= 2). Evaluated as independent source cards to prevent false merging.",
                })

    return groups, ambiguous_pairs
