"""Read-only loading of one split in the verified ``<split>/images`` + ``<split>/labels`` layout."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from idocr.data.audit.labels import LabelIssue, YoloAnnotation, parse_yolo_file
from idocr.data.audit.naming import roboflow_source_stem

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class LabeledImage:
    split: str
    image_path: Path
    label_path: Path | None
    annotations: list[YoloAnnotation] = field(default_factory=list)
    issues: list[LabelIssue] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return self.image_path.name

    @property
    def source(self) -> str:
        """Original-upload name (Roboflow) — augmented copies share it."""
        stem = self.image_path.stem
        return roboflow_source_stem(stem) or stem

    @property
    def class_counts(self) -> Counter[int]:
        return Counter(a.class_id for a in self.annotations)

    @property
    def annotation_types(self) -> list[str]:
        return sorted({a.kind for a in self.annotations})


def load_split(dataset_root: Path, split: str) -> list[LabeledImage]:
    """Parse every image/label pair of one split. Files are only read."""
    images_dir, labels_dir = dataset_root / split / "images", dataset_root / split / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No such split: {images_dir}")
    items = []
    for img in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        label = labels_dir / f"{img.stem}.txt"
        rel = f"{split}/labels/{label.name}"
        if not label.is_file():
            items.append(LabeledImage(split, img, None, issues=[LabelIssue(rel, "missing_label_file")]))
            continue
        parsed = parse_yolo_file(label, rel)
        items.append(LabeledImage(split, img, label, parsed.annotations, parsed.issues))
    return items
