"""Builder for common multi-document YOLO field-detector training datasets."""

from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from idocr.data.annotation_schema import (
    DETECTOR_CLASSES,
    FIELD_NAME_TO_DETECTOR_CLASS,
    AnnotatedDocument,
    get_detector_class_id,
)
from idocr.data.converters.yolo import YoloDatasetConverter
from idocr.types import DocumentType


@dataclass
class DetectorDatasetStats:
    """Detailed statistics for the generated detector dataset."""
    total_images: int = 0
    images_by_split: dict[str, int] = field(default_factory=dict)
    images_by_dataset: dict[str, int] = field(default_factory=dict)
    total_targets: int = 0
    targets_by_class: dict[int, int] = field(default_factory=dict)
    targets_by_dataset: dict[str, int] = field(default_factory=dict)
    targets_by_split: dict[str, int] = field(default_factory=dict)
    excluded_class_4_count: int = 0
    empty_label_images: int = 0
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_images": self.total_images,
            "images_by_split": dict(self.images_by_split),
            "images_by_dataset": dict(self.images_by_dataset),
            "total_targets": self.total_targets,
            "targets_by_class": {
                f"{cid} ({DETECTOR_CLASSES.get(cid, 'unknown')})": count
                for cid, count in sorted(self.targets_by_class.items())
            },
            "targets_by_dataset": dict(self.targets_by_dataset),
            "targets_by_split": dict(self.targets_by_split),
            "excluded_class_4_count": self.excluded_class_4_count,
            "empty_label_images": self.empty_label_images,
            "validation_errors": list(self.validation_errors),
        }


class DetectorDatasetBuilder:
    """Builds a unified YOLO detector dataset combining Aadhaar and PAN."""

    def __init__(
        self,
        output_dir: str | Path = "data/processed/detector",
        source_mapping_config: str | Path = "configs/detection_classes.yaml",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.source_mapping_config = Path(source_mapping_config)

        # Load mapping from config
        self.source_to_detector: dict[str, dict[int, int | None]] = {
            "aadhaar": {0: 3, 1: 1, 2: 2, 3: 0, 4: None},
            "pan": {0: 1, 1: 5, 2: 0, 3: 4},
        }
        if self.source_mapping_config.exists():
            with self.source_mapping_config.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if "source_to_detector_mapping" in cfg:
                    self.source_to_detector = cfg["source_to_detector_mapping"]

    def build_dataset(
        self,
        aadhaar_processed_dir: str | Path = "data/processed/aadhaar",
        pan_processed_dir: str | Path = "data/processed/pan",
        splits: Sequence[str] = ("train", "valid", "test"),
        overwrite: bool = True,
    ) -> DetectorDatasetStats:
        """Generate detector dataset combining processed Aadhaar and PAN splits."""
        aadhaar_dir = Path(aadhaar_processed_dir)
        pan_dir = Path(pan_processed_dir)
        stats = DetectorDatasetStats()

        for split in splits:
            img_out = self.output_dir / split / "images"
            lbl_out = self.output_dir / split / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            # Process Aadhaar
            if (aadhaar_dir / split).exists():
                self._process_dataset_split(
                    dataset_name="aadhaar",
                    src_split_dir=aadhaar_dir / split,
                    img_out=img_out,
                    lbl_out=lbl_out,
                    split_name=split,
                    stats=stats,
                )

            # Process PAN
            if (pan_dir / split).exists():
                self._process_dataset_split(
                    dataset_name="pan",
                    src_split_dir=pan_dir / split,
                    img_out=img_out,
                    lbl_out=lbl_out,
                    split_name=split,
                    stats=stats,
                )

        # Write data.yaml descriptor for YOLO
        self._write_data_yaml()

        return stats

    def _process_dataset_split(
        self,
        dataset_name: str,
        src_split_dir: Path,
        img_out: Path,
        lbl_out: Path,
        split_name: str,
        stats: DetectorDatasetStats,
    ) -> None:
        src_imgs = sorted((src_split_dir / "images").glob("*.*"))
        mapping = self.source_to_detector.get(dataset_name, {})

        for img_path in src_imgs:
            stats.total_images += 1
            stats.images_by_split[split_name] = stats.images_by_split.get(split_name, 0) + 1
            stats.images_by_dataset[dataset_name] = stats.images_by_dataset.get(dataset_name, 0) + 1

            # Copy image if not present or size changed
            dst_img = img_out / img_path.name
            if not dst_img.exists() or dst_img.stat().st_size != img_path.stat().st_size:
                shutil.copy2(img_path, dst_img)

            # Read source label file
            lbl_path = src_split_dir / "labels" / f"{img_path.stem}.txt"
            dst_lbl = lbl_out / f"{img_path.stem}.txt"

            detector_lines: list[str] = []

            if lbl_path.exists():
                raw_lines = [l.strip() for l in lbl_path.read_text("utf-8-sig").splitlines() if l.strip()]
                for line in raw_lines:
                    tokens = line.split()
                    if not tokens:
                        continue

                    try:
                        src_cls = int(tokens[0])
                        coords = [float(x) for x in tokens[1:]]
                    except ValueError as exc:
                        stats.validation_errors.append(f"{img_path.name}: Malformed line '{line}': {exc}")
                        continue

                    # Check mapping
                    if src_cls not in mapping:
                        stats.validation_errors.append(
                            f"{img_path.name}: Unmapped source class {src_cls} in {dataset_name}"
                        )
                        continue

                    det_cls = mapping[src_cls]

                    # If explicitly excluded (e.g. Class 4 in Aadhaar)
                    if det_cls is None:
                        if dataset_name == "aadhaar" and src_cls == 4:
                            stats.excluded_class_4_count += 1
                        continue

                    # Ensure coordinates are a valid 4-token box
                    if len(coords) != 4:
                        stats.validation_errors.append(
                            f"{img_path.name}: Non-box geometry ({len(coords)} tokens) for detector class {det_cls}: '{line}'"
                        )
                        continue

                    cx, cy, w, h = coords

                    # Validate bounds [0, 1]
                    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        stats.validation_errors.append(
                            f"{img_path.name}: Out of bounds coordinates: cx={cx}, cy={cy}, w={w}, h={h}"
                        )
                        continue

                    # Format standard YOLO label line: <class_id> <cx> <cy> <w> <h>
                    line_str = f"{det_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    detector_lines.append(line_str)

                    stats.total_targets += 1
                    stats.targets_by_class[det_cls] = stats.targets_by_class.get(det_cls, 0) + 1
                    stats.targets_by_dataset[dataset_name] = stats.targets_by_dataset.get(dataset_name, 0) + 1
                    stats.targets_by_split[split_name] = stats.targets_by_split.get(split_name, 0) + 1

            # Write label file (empty file if 0 targets)
            dst_lbl.write_text("\n".join(detector_lines) + ("\n" if detector_lines else ""), encoding="utf-8")
            if not detector_lines:
                stats.empty_label_images += 1

    def _write_data_yaml(self) -> Path:
        """Write YOLO dataset descriptor file."""
        data_yaml_path = self.output_dir / "data.yaml"
        yaml_content = {
            "path": ".",
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "names": {cid: name for cid, name in sorted(DETECTOR_CLASSES.items())},
        }
        with data_yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        return data_yaml_path
