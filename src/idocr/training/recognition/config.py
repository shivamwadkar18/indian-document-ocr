"""Recognizer training configuration (YAML -> validated dataclass).

Configs live in ``configs/recognition/``. Unknown keys are rejected.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from idocr.data.augmentation.ocr import OcrAugmentConfig
from idocr.utils.config import load_config

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RecognizerConfigError(ValueError):
    pass


@dataclass
class RecognizerTrainConfig:
    experiment_name: str
    dataset_dir: Path = Path("data/processed/recognition")
    output_dir: Path = Path("experiments/runs")
    epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    num_workers: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:0" ...
    amp: bool = False  # mixed precision; CUDA only
    seed: int = 42
    # Optional caps (first N records of each split; records are field-interleaved
    # so a cap that is a multiple of 6 stays balanced). null = whole split.
    max_train_samples: int | None = None
    max_valid_samples: int | None = None
    hidden_size: int = 128
    lstm_layers: int = 2
    dropout: float = 0.1
    log_every: int = 50
    # Optional train-split augmentation (OcrAugmentConfig fields); {} or
    # enabled: false keeps the clean synthetic baseline.
    augmentation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)
        self.output_dir = Path(self.output_dir)
        errors = []
        if not isinstance(self.experiment_name, str) or not _NAME_RE.match(self.experiment_name):
            errors.append(f"experiment_name must match {_NAME_RE.pattern!r}, got {self.experiment_name!r}")
        for name in ("epochs", "batch_size", "hidden_size", "lstm_layers", "log_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                errors.append(f"{name} must be a positive integer, got {value!r}")
        for name in ("num_workers", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"{name} must be a non-negative integer, got {value!r}")
        for name in ("max_train_samples", "max_valid_samples"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                errors.append(f"{name} must be null or a positive integer, got {value!r}")
        if not 0 < self.lr <= 1:
            errors.append(f"lr must be in (0, 1], got {self.lr!r}")
        if self.weight_decay < 0 or self.grad_clip < 0:
            errors.append("weight_decay and grad_clip must be non-negative")
        if not 0 <= self.dropout < 1:
            errors.append(f"dropout must be in [0, 1), got {self.dropout!r}")
        if not isinstance(self.device, str) or not re.fullmatch(r"auto|cpu|cuda(:\d+)?", self.device):
            errors.append(f"device must be 'auto', 'cpu' or 'cuda[:N]', got {self.device!r}")
        try:
            OcrAugmentConfig.from_mapping(self.augmentation)
        except (TypeError, ValueError) as exc:
            errors.append(f"augmentation: {exc}")
        if errors:
            raise RecognizerConfigError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dataset_dir"] = self.dataset_dir.as_posix()
        d["output_dir"] = self.output_dir.as_posix()
        return d

    @property
    def augment_config(self) -> OcrAugmentConfig:
        return OcrAugmentConfig.from_mapping(self.augmentation)

    def run_dir(self, root: Path) -> Path:
        out = self.output_dir if self.output_dir.is_absolute() else root / self.output_dir
        return (out / self.experiment_name).resolve()


def config_from_dict(data: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> RecognizerTrainConfig:
    merged = {**data, **{k: v for k, v in (overrides or {}).items() if v is not None}}
    unknown = sorted(set(merged) - {f.name for f in fields(RecognizerTrainConfig)})
    if unknown:
        raise RecognizerConfigError(f"Unknown config keys: {unknown}")
    if "experiment_name" not in merged:
        raise RecognizerConfigError("experiment_name is required")
    return RecognizerTrainConfig(**merged)


def load_recognizer_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> RecognizerTrainConfig:
    """Load and validate a recognizer config YAML; ``overrides`` (non-None values) win."""
    return config_from_dict(load_config(path), overrides)
