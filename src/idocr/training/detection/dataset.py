"""Detector dataset checks and run-local ``data.yaml`` resolution.

The dataset's own ``data.yaml`` uses ``path: .``. Ultralytics resolves a
relative ``path`` against the *current working directory*, not the yaml's
folder, so training writes a resolved copy (absolute path) into the run
directory instead of relying on the cwd. The dataset itself is never written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from idocr.data.annotation_schema import DETECTOR_CLASSES

REQUIRED_SPLITS = ("train", "val")


class DetectorDatasetError(ValueError):
    pass


@dataclass
class ResolvedDataset:
    yaml_path: Path        # original data.yaml
    root: Path             # absolute dataset root
    names: dict[int, str]
    splits: dict[str, Path]  # ultralytics split key -> absolute images dir

    def to_ultralytics(self) -> dict[str, Any]:
        out: dict[str, Any] = {"path": self.root.as_posix(), "names": dict(self.names), "nc": len(self.names)}
        for key, images_dir in self.splits.items():
            out[key] = images_dir.relative_to(self.root).as_posix()
        return out


def resolve_dataset(yaml_path: str | Path) -> ResolvedDataset:
    """Load a detector data.yaml and check paths and the locked class vocabulary."""
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.is_file():
        raise DetectorDatasetError(f"Dataset descriptor not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise DetectorDatasetError(f"{yaml_path} must contain a mapping")

    rel_root = Path(data.get("path") or ".")
    root = rel_root if rel_root.is_absolute() else (yaml_path.parent / rel_root).resolve()
    if not root.is_dir():
        raise DetectorDatasetError(f"Dataset root does not exist: {root}")

    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise DetectorDatasetError("data.yaml 'names' must be a mapping or list")
    names = {int(k): str(v) for k, v in names.items()}
    if "nc" in data and int(data["nc"]) != len(names):
        raise DetectorDatasetError(f"nc={data['nc']} does not match {len(names)} names")
    if names != DETECTOR_CLASSES:
        raise DetectorDatasetError(
            f"Class vocabulary mismatch: expected {DETECTOR_CLASSES} ({len(DETECTOR_CLASSES)} classes), got {names}"
        )

    splits: dict[str, Path] = {}
    for key in ("train", "val", "test"):
        if key not in data:
            if key in REQUIRED_SPLITS:
                raise DetectorDatasetError(f"data.yaml is missing required split '{key}'")
            continue
        images_dir = (root / str(data[key])).resolve()
        labels_dir = images_dir.parent / "labels"
        if not images_dir.is_dir():
            raise DetectorDatasetError(f"Split '{key}' images dir not found: {images_dir}")
        if not labels_dir.is_dir():
            raise DetectorDatasetError(f"Split '{key}' labels dir not found: {labels_dir}")
        if not any(images_dir.iterdir()):
            raise DetectorDatasetError(f"Split '{key}' has no images: {images_dir}")
        splits[key] = images_dir
    return ResolvedDataset(yaml_path, root, names, splits)


def write_resolved_yaml(dataset: ResolvedDataset, run_dir: Path) -> Path:
    path = run_dir / "data.resolved.yaml"
    path.write_text(yaml.safe_dump(dataset.to_ultralytics(), sort_keys=False), encoding="utf-8")
    return path
