"""Field-detector training with Ultralytics YOLO.

Pipeline: validate config -> check dataset (+ optional full validator) ->
create a new run dir (never overwrites) -> write resolved data.yaml and
run_config.yaml -> train -> validate best checkpoint on the val split ->
write metrics.json + update run_config.yaml.

Run layout (``<output_dir>/<experiment_name>/``)::

    run_config.yaml       resolved config, environment, dataset, status
    data.resolved.yaml    data.yaml copy with an absolute dataset path
    weights/best.pt, weights/last.pt, args.yaml, results.csv   (Ultralytics)
    val/                  final validation outputs
    metrics.json          overall + per-class P / R / mAP50 / mAP50-95

Ultralytics is imported lazily so the rest of the package (and the tests)
work without it.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from idocr.data.annotation_schema import DETECTOR_CLASSES
from idocr.training.detection.config import DetectorTrainConfig, config_from_dict
from idocr.training.detection.dataset import ResolvedDataset, resolve_dataset, write_resolved_yaml
from idocr.training.detection.metrics import extract_detection_metrics
from idocr.utils.paths import find_project_root

RUN_CONFIG_FILENAME = "run_config.yaml"
METRICS_FILENAME = "metrics.json"


class RunExistsError(FileExistsError):
    pass


def prepare_run_dir(config: DetectorTrainConfig, root: Path) -> Path:
    """Create a fresh run directory; refuse to reuse an existing one."""
    run_dir = config.run_dir(root)
    if run_dir.exists():
        raise RunExistsError(
            f"Run directory already exists: {run_dir}. Choose a new experiment_name; runs are never overwritten."
        )
    run_dir.mkdir(parents=True)
    return run_dir


def ultralytics_train_args(config: DetectorTrainConfig, data_yaml: Path, run_dir: Path) -> dict[str, Any]:
    """Keyword arguments for ``YOLO.train``. Managed keys cannot be overridden by ``train_args``."""
    return {
        **config.train_args,
        "data": str(data_yaml),
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,  # dir was just created by prepare_run_dir
        "pretrained": config.pretrained,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "epochs": config.epochs,
        "patience": config.patience,
        "workers": config.workers,
        "device": config.device,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "fraction": config.fraction,
        "plots": config.plots,
        "val": True,
    }


@contextlib.contextmanager
def no_dataset_cache_writes() -> Iterator[None]:
    """Stop Ultralytics writing ``labels.cache`` files next to the dataset labels.

    Labels are re-scanned each run instead (seconds for this dataset), which
    keeps the detector dataset byte-for-byte unchanged.
    """
    import ultralytics.data.dataset as ul_dataset

    def stamp_version_only(prefix: str, path: Path, x: dict, version: str) -> None:
        # Keep the in-memory side effect Ultralytics relies on (get_labels pops
        # "version"); skip only the disk write.
        x["version"] = version

    original = ul_dataset.save_dataset_cache_file
    ul_dataset.save_dataset_cache_file = stamp_version_only
    try:
        yield
    finally:
        ul_dataset.save_dataset_cache_file = original


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "torchvision", "ultralytics"):
        try:
            env[mod] = str(__import__(mod).__version__)  # torch's TorchVersion is not YAML-safe
        except Exception:  # noqa: BLE001
            env[mod] = None
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        env["cuda_available"] = None
    return env


def _write_run_config(run_dir: Path, payload: Mapping[str, Any]) -> None:
    (run_dir / RUN_CONFIG_FILENAME).write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _validate_dataset_full(dataset: ResolvedDataset, root: Path) -> dict[str, Any]:
    from idocr.data.detection import DetectorDatasetValidator

    result = DetectorDatasetValidator(
        dataset_dir=dataset.root,
        manifests_dir=root / "data" / "processed" / "manifests",
        raw_dir=root / "data" / "raw",
        processed_dir=root / "data" / "processed",
    ).validate()
    if not result.is_valid:
        raise ValueError(f"Detector dataset failed validation: {result.errors[:10]}")
    return {"is_valid": True, "total_images": result.total_images, "total_targets": result.total_targets,
            "warnings": result.warnings[:20]}


def train(config: DetectorTrainConfig | Mapping[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    """Train and validate a field detector. Returns the final run record."""
    cfg = config if isinstance(config, DetectorTrainConfig) else config_from_dict(config)
    root_path = Path(root).resolve() if root is not None else find_project_root()
    dataset_yaml = cfg.dataset if cfg.dataset.is_absolute() else root_path / cfg.dataset
    dataset = resolve_dataset(dataset_yaml)
    dataset_check = _validate_dataset_full(dataset, root_path) if cfg.validate_dataset else {"skipped": True}

    run_dir = prepare_run_dir(cfg, root_path)
    data_yaml = write_resolved_yaml(dataset, run_dir)
    record: dict[str, Any] = {
        "experiment_name": cfg.experiment_name,
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "dataset": {"descriptor": dataset.yaml_path.as_posix(), "root": dataset.root.as_posix(),
                    "classes": {int(k): v for k, v in dataset.names.items()}, "preflight": dataset_check},
        "environment": _environment(),
    }
    _write_run_config(run_dir, record)

    if cfg.offline:
        os.environ["YOLO_OFFLINE"] = "1"  # must precede the ultralytics import
    from ultralytics import YOLO

    try:
        with no_dataset_cache_writes():
            model = YOLO(cfg.model)
            model.train(**ultralytics_train_args(cfg, data_yaml, run_dir))
            best = run_dir / "weights" / "best.pt"
            checkpoint = best if best.is_file() else run_dir / "weights" / "last.pt"
            val_metrics = YOLO(str(checkpoint)).val(
                data=str(data_yaml), split="val", imgsz=cfg.imgsz, batch=cfg.batch, device=cfg.device,
                workers=cfg.workers, plots=cfg.plots, project=str(run_dir), name="val", exist_ok=True,
            )
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      finished_at=datetime.now().isoformat(timespec="seconds"))
        _write_run_config(run_dir, record)
        raise

    metrics = extract_detection_metrics(val_metrics, DETECTOR_CLASSES)
    metrics.update(split="val", checkpoint=checkpoint.relative_to(run_dir).as_posix())
    (run_dir / METRICS_FILENAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    record.update(
        status="completed",
        finished_at=datetime.now().isoformat(timespec="seconds"),
        checkpoints={p.name: p.relative_to(run_dir).as_posix() for p in sorted((run_dir / "weights").glob("*.pt"))},
        metrics_file=METRICS_FILENAME,
    )
    _write_run_config(run_dir, record)
    return record
