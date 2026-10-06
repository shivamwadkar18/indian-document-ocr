"""Field-detector training (Ultralytics YOLO). See ``train.py`` for the pipeline."""

from idocr.training.detection.config import (
    DetectorConfigError,
    DetectorTrainConfig,
    config_from_dict,
    load_detector_config,
)
from idocr.training.detection.dataset import DetectorDatasetError, ResolvedDataset, resolve_dataset
from idocr.training.detection.metrics import extract_detection_metrics
from idocr.training.detection.train import RunExistsError, prepare_run_dir, train, ultralytics_train_args

__all__ = [
    "DetectorConfigError",
    "DetectorDatasetError",
    "DetectorTrainConfig",
    "ResolvedDataset",
    "RunExistsError",
    "config_from_dict",
    "extract_detection_metrics",
    "load_detector_config",
    "prepare_run_dir",
    "resolve_dataset",
    "train",
    "ultralytics_train_args",
]
