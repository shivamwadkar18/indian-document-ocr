"""Detector training configuration (YAML -> validated dataclass).

Configs live in ``configs/detector/``. Unknown keys are rejected so typos do
not silently fall back to framework defaults; framework-specific extras go
under ``train_args`` and are passed through to Ultralytics unchanged.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from idocr.utils.config import load_config

#: Allowed optimizer names (Ultralytics vocabulary).
OPTIMIZERS = {"SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class DetectorConfigError(ValueError):
    pass


@dataclass
class DetectorTrainConfig:
    experiment_name: str
    dataset: Path = Path("data/processed/detector/data.yaml")
    # "yolov8n.yaml" builds the architecture from scratch (no download);
    # "yolov8n.pt" / a local .pt path starts from pretrained weights.
    model: str = "yolov8n.yaml"
    pretrained: bool = False
    imgsz: int = 640
    batch: int = 8
    epochs: int = 1
    patience: int = 10
    workers: int = 2
    device: str = "cpu"
    optimizer: str = "SGD"
    lr0: float = 0.01
    seed: int = 42
    deterministic: bool = True
    output_dir: Path = Path("experiments/runs")
    # Fraction of the training split to use (1.0 = all). Smoke tests may lower it.
    fraction: float = 1.0
    # Ultralytics plots render training images (identity documents) into the run dir.
    plots: bool = False
    # Block network access (analytics, font/weight downloads) via YOLO_OFFLINE.
    offline: bool = True
    # Run the full DetectorDatasetValidator before training.
    validate_dataset: bool = True
    # Passed through to Ultralytics train() (augmentation etc.).
    train_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset = Path(self.dataset)
        self.output_dir = Path(self.output_dir)
        self.validate()

    def validate(self) -> None:
        errors = []
        if not isinstance(self.experiment_name, str) or not _NAME_RE.match(self.experiment_name):
            errors.append(f"experiment_name must match {_NAME_RE.pattern!r}, got {self.experiment_name!r}")
        if not str(self.model).endswith((".yaml", ".yml", ".pt")):
            errors.append(f"model must be a .yaml architecture or .pt weights file, got {self.model!r}")
        for name in ("imgsz", "batch", "epochs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                errors.append(f"{name} must be a positive integer, got {value!r}")
        for name in ("patience", "workers", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"{name} must be a non-negative integer, got {value!r}")
        if self.imgsz % 32:
            errors.append(f"imgsz must be a multiple of 32 (YOLO stride), got {self.imgsz}")
        if self.optimizer not in OPTIMIZERS:
            errors.append(f"optimizer must be one of {sorted(OPTIMIZERS)}, got {self.optimizer!r}")
        if not isinstance(self.lr0, (int, float)) or isinstance(self.lr0, bool) or not 0 < self.lr0 <= 1:
            errors.append(f"lr0 must be in (0, 1], got {self.lr0!r}")
        if not isinstance(self.fraction, (int, float)) or not 0 < self.fraction <= 1:
            errors.append(f"fraction must be in (0, 1], got {self.fraction!r}")
        if not isinstance(self.device, str) or not self.device:
            errors.append(f"device must be a non-empty string such as 'cpu' or '0', got {self.device!r}")
        reserved = {f.name for f in fields(self)} | {"data", "project", "name", "exist_ok", "resume"}
        clash = sorted(set(self.train_args) & reserved)
        if clash:
            errors.append(f"train_args may not override managed keys: {clash}")
        if errors:
            raise DetectorConfigError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dataset"] = self.dataset.as_posix()
        d["output_dir"] = self.output_dir.as_posix()
        return d

    def run_dir(self, root: Path) -> Path:
        out = self.output_dir if self.output_dir.is_absolute() else root / self.output_dir
        return (out / self.experiment_name).resolve()


def config_from_dict(data: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> DetectorTrainConfig:
    merged = {**data, **{k: v for k, v in (overrides or {}).items() if v is not None}}
    known = {f.name for f in fields(DetectorTrainConfig)}
    unknown = sorted(set(merged) - known)
    if unknown:
        raise DetectorConfigError(f"Unknown config keys: {unknown}")
    if "experiment_name" not in merged:
        raise DetectorConfigError("experiment_name is required")
    return DetectorTrainConfig(**merged)


def load_detector_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> DetectorTrainConfig:
    """Load and validate a detector config YAML; ``overrides`` (non-None values) win."""
    return config_from_dict(load_config(path), overrides)
