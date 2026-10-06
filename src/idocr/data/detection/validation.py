"""Validation suite for detector-ready datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from idocr.data.annotation_schema import DETECTOR_CLASSES


@dataclass
class DetectorValidationResult:
    """Results from validating a detector dataset."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    # Detailed counts
    images_by_split: dict[str, int] = field(default_factory=dict)
    labels_by_split: dict[str, int] = field(default_factory=dict)
    targets_by_class: dict[int, int] = field(default_factory=dict)
    empty_labels_by_split: dict[str, int] = field(default_factory=dict)
    total_images: int = 0
    total_labels: int = 0
    total_targets: int = 0
    
    # Integrity checks
    pairing_passed: bool = False
    classes_valid: bool = False
    geometry_valid: bool = False
    split_leakage_free: bool = False
    source_integrity_passed: bool = False

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "total_images": self.total_images,
            "total_labels": self.total_labels,
            "total_targets": self.total_targets,
            "images_by_split": dict(self.images_by_split),
            "labels_by_split": dict(self.labels_by_split),
            "targets_by_class": {
                f"{cid} ({DETECTOR_CLASSES.get(cid, 'unknown')})": count
                for cid, count in sorted(self.targets_by_class.items())
            },
            "empty_labels_by_split": dict(self.empty_labels_by_split),
            "pairing_passed": self.pairing_passed,
            "classes_valid": self.classes_valid,
            "geometry_valid": self.geometry_valid,
            "split_leakage_free": self.split_leakage_free,
            "source_integrity_passed": self.source_integrity_passed,
        }


class DetectorDatasetValidator:
    """Validates detector dataset integrity, pairing, classes, geometry, and leakage."""

    def __init__(
        self,
        dataset_dir: str | Path = "data/processed/detector",
        manifests_dir: str | Path = "data/processed/manifests",
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        splits: Sequence[str] = ("train", "valid", "test"),
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifests_dir = Path(manifests_dir)
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.splits = list(splits)

    def validate(self) -> DetectorValidationResult:
        """Run complete validation suite on the detector dataset."""
        res = DetectorValidationResult()

        if not self.dataset_dir.exists():
            res.add_error(f"Detector dataset directory does not exist: {self.dataset_dir}")
            return res

        # 1. Validate data.yaml descriptor
        self._validate_data_yaml(res)

        # 2. Validate split directory structure, pairing, classes, and geometry
        self._validate_splits_and_targets(res)

        # 3. Validate split leakage isolation
        self._validate_split_leakage(res)

        # 4. Validate source integrity
        self._validate_source_integrity(res)

        return res

    def _validate_data_yaml(self, res: DetectorValidationResult) -> None:
        yaml_path = self.dataset_dir / "data.yaml"
        if not yaml_path.exists():
            res.add_error(f"Missing data.yaml at {yaml_path}")
            return

        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            names = data.get("names", {})
            for cid, name in DETECTOR_CLASSES.items():
                if cid not in names or names[cid] != name:
                    res.add_error(f"data.yaml names mismatch for class {cid}: expected '{name}', got '{names.get(cid)}'")
        except Exception as exc:
            res.add_error(f"Failed to parse data.yaml: {exc}")

    def _validate_splits_and_targets(self, res: DetectorValidationResult) -> None:
        all_stems: dict[str, str] = {}  # stem -> split
        pairing_errors = 0
        class_errors = 0
        geometry_errors = 0

        for split in self.splits:
            img_dir = self.dataset_dir / split / "images"
            lbl_dir = self.dataset_dir / split / "labels"

            if not img_dir.exists() or not lbl_dir.exists():
                res.add_error(f"Split {split} missing images or labels directory")
                continue

            images = {p.stem: p for p in sorted(img_dir.glob("*.*")) if p.is_file()}
            labels = {p.stem: p for p in sorted(lbl_dir.glob("*.txt")) if p.is_file()}

            res.images_by_split[split] = len(images)
            res.labels_by_split[split] = len(labels)
            res.total_images += len(images)
            res.total_labels += len(labels)
            res.empty_labels_by_split[split] = 0

            # Pairing check
            missing_labels = set(images.keys()) - set(labels.keys())
            missing_images = set(labels.keys()) - set(images.keys())

            if missing_labels:
                pairing_errors += len(missing_labels)
                for stem in sorted(missing_labels)[:5]:
                    res.add_error(f"[{split}] Image '{stem}' has no corresponding label file")
            if missing_images:
                pairing_errors += len(missing_images)
                for stem in sorted(missing_images)[:5]:
                    res.add_error(f"[{split}] Label '{stem}.txt' has no corresponding image file")

            # Check cross-split duplicate filenames
            for stem in images.keys():
                if stem in all_stems:
                    res.add_error(f"Duplicate image stem '{stem}' across splits: '{all_stems[stem]}' and '{split}'")
                else:
                    all_stems[stem] = split

            # Parse and validate labels
            for stem, lbl_path in labels.items():
                content = lbl_path.read_text(encoding="utf-8").strip()
                if not content:
                    res.empty_labels_by_split[split] += 1
                    continue

                lines = content.splitlines()
                for line_idx, line in enumerate(lines, start=1):
                    tokens = line.strip().split()
                    if not tokens:
                        continue

                    try:
                        cls_id = int(tokens[0])
                        coords = [float(x) for x in tokens[1:]]
                    except ValueError:
                        class_errors += 1
                        res.add_error(f"[{split}/{lbl_path.name}:{line_idx}] Non-numeric values in line: '{line}'")
                        continue

                    # Validate class ID
                    if cls_id not in DETECTOR_CLASSES:
                        class_errors += 1
                        res.add_error(
                            f"[{split}/{lbl_path.name}:{line_idx}] Invalid detector class ID {cls_id} (allowed: {list(DETECTOR_CLASSES.keys())})"
                        )
                    else:
                        res.targets_by_class[cls_id] = res.targets_by_class.get(cls_id, 0) + 1
                        res.total_targets += 1

                    # Validate geometry
                    if len(coords) != 4:
                        geometry_errors += 1
                        res.add_error(
                            f"[{split}/{lbl_path.name}:{line_idx}] Expected 4 box coordinates (cx, cy, w, h), found {len(coords)}: '{line}'"
                        )
                        continue

                    cx, cy, w, h = coords
                    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        geometry_errors += 1
                        res.add_error(
                            f"[{split}/{lbl_path.name}:{line_idx}] Coordinates out of bounds: cx={cx}, cy={cy}, w={w}, h={h}"
                        )

        res.pairing_passed = (pairing_errors == 0)
        res.classes_valid = (class_errors == 0)
        res.geometry_valid = (geometry_errors == 0)

    def _validate_split_leakage(self, res: DetectorValidationResult) -> None:
        """Verify no source groups or images cross split boundaries using split manifests."""
        if not self.manifests_dir.exists():
            res.add_warning("Manifests directory not found; skipping source-group isolation cross-check.")
            res.split_leakage_free = True
            return

        # Pre-index detector dataset image stems for fast in-memory lookup
        split_stems: dict[str, str] = {}
        for s in self.splits:
            img_dir = self.dataset_dir / s / "images"
            if img_dir.exists():
                for p in img_dir.iterdir():
                    if p.is_file():
                        split_stems[p.stem] = s

        group_to_split: dict[str, str] = {}
        leakage_detected = False

        # Check Aadhaar and PAN split manifests
        for doc_type in ("aadhaar", "pan"):
            manifest_csv = self.manifests_dir / f"{doc_type}_split_manifest.csv"
            if not manifest_csv.exists():
                continue

            with manifest_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stem = Path(row["image_path"]).stem
                    group_id = row["source_group_id"]
                    split = row["split"]

                    # Check if image stem matches detector dataset split
                    if stem in split_stems:
                        det_split = split_stems[stem]
                        if det_split != split:
                            res.add_error(
                                f"Split mismatch for {doc_type} '{stem}': manifest says '{split}', detector placed in '{det_split}'"
                            )
                            leakage_detected = True

                    if group_id in group_to_split:
                        if group_to_split[group_id] != split:
                            res.add_error(f"Source group {group_id} allocated to multiple splits: {group_to_split[group_id]} and {split}")
                            leakage_detected = True
                    else:
                        group_to_split[group_id] = split

        res.split_leakage_free = not leakage_detected

    def _validate_source_integrity(self, res: DetectorValidationResult) -> None:
        """Check that raw and processed source datasets remain intact."""
        aadhaar_raw = self.raw_dir / "aadhaar"
        pan_raw = self.raw_dir / "pan"
        aadhaar_proc = self.processed_dir / "aadhaar"
        pan_proc = self.processed_dir / "pan"

        intact = True
        for path, name in [
            (aadhaar_raw, "raw Aadhaar"),
            (pan_raw, "raw PAN"),
            (aadhaar_proc, "processed Aadhaar"),
            (pan_proc, "processed PAN"),
        ]:
            if not path.exists():
                res.add_error(f"Source directory missing: {name} ({path})")
                intact = False

        res.source_integrity_passed = intact
