"""Split layout detection.

Verified layout (Aadhaar and PAN, Roboflow YOLO export)::

    <dataset>/<split>/images/<stem>.<image ext>
    <dataset>/<split>/labels/<stem>.txt

Only this layout is recognised. Datasets that do not follow it produce no
splits, and the runner falls back to whole-dataset analysis with the
unverified stem heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from idocr.data.discovery import IGNORED_FILENAMES

IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"


@dataclass
class SplitLayout:
    name: str
    root: Path
    images_dir: Path
    labels_dir: Path
    name_is_known_split: bool  # name matches a configured split-name candidate

    def to_dict(self, dataset_root: Path) -> dict:
        return {
            "name": self.name,
            "images_dir": self.images_dir.relative_to(dataset_root).as_posix(),
            "labels_dir": self.labels_dir.relative_to(dataset_root).as_posix(),
            "name_is_known_split": self.name_is_known_split,
        }


@dataclass
class LayoutResult:
    splits: list[SplitLayout] = field(default_factory=list)
    # paths (relative to dataset root) that are not part of any detected split
    unexpected: list[str] = field(default_factory=list)


def detect_layout(root: Path, split_name_candidates: Iterable[str] = ()) -> LayoutResult:
    known = {s.lower() for s in split_name_candidates}
    result = LayoutResult()
    for child in sorted(root.iterdir()):
        if child.name in IGNORED_FILENAMES:
            continue
        images, labels = child / IMAGES_DIRNAME, child / LABELS_DIRNAME
        if child.is_dir() and images.is_dir() and labels.is_dir():
            result.splits.append(SplitLayout(child.name, child, images, labels, child.name.lower() in known))
            for extra in sorted(child.iterdir()):
                if extra.name not in (IMAGES_DIRNAME, LABELS_DIRNAME) and extra.name not in IGNORED_FILENAMES:
                    result.unexpected.append(extra.relative_to(root).as_posix())
            for sub in (images, labels):
                for nested in sorted(sub.iterdir()):
                    if nested.is_dir():
                        result.unexpected.append(nested.relative_to(root).as_posix())
        else:
            result.unexpected.append(child.relative_to(root).as_posix())
    return result
