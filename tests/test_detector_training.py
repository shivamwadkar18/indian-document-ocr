"""Detector training pipeline tests. No real training: Ultralytics is mocked."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from idocr.data.annotation_schema import DETECTOR_CLASSES
from idocr.training.detection import (
    DetectorConfigError,
    DetectorDatasetError,
    DetectorTrainConfig,
    RunExistsError,
    config_from_dict,
    extract_detection_metrics,
    load_detector_config,
    prepare_run_dir,
    resolve_dataset,
    train,
    ultralytics_train_args,
)
from idocr.training.detection.dataset import write_resolved_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    """Minimal detector layout (empty placeholder image files; never decoded here)."""
    root = tmp_path / "detector"
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / "a.jpg").write_bytes(b"")
    data = {"path": ".", "train": "train/images", "val": "valid/images", "test": "test/images",
            "names": dict(DETECTOR_CLASSES)}
    (root / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return root


def _cfg(**kw):
    return DetectorTrainConfig(**{"experiment_name": "t", **kw})


# --- configuration ----------------------------------------------------------


@pytest.mark.parametrize("name", ["smoke_test_yolov8n.yaml", "baseline_yolov8n.yaml"])
def test_repo_configs_load(name):
    cfg = load_detector_config(REPO_ROOT / "configs" / "detector" / name)
    assert cfg.seed == 42 and cfg.imgsz == 640 and cfg.train_args["fliplr"] == 0.0


def test_smoke_config_defaults():
    cfg = load_detector_config(REPO_ROOT / "configs" / "detector" / "smoke_test_yolov8n.yaml")
    assert (cfg.model, cfg.epochs, cfg.batch, cfg.workers, cfg.device) == ("yolov8n.yaml", 1, 8, 2, "cpu")
    assert cfg.experiment_name == "smoke_test_yolov8n" and cfg.offline and not cfg.plots


def test_baseline_config_documents_real_run():
    cfg = load_detector_config(REPO_ROOT / "configs" / "detector" / "baseline_yolov8n.yaml")
    assert (cfg.epochs, cfg.patience) == (50, 10)
    assert cfg.experiment_name != "smoke_test_yolov8n"


def test_overrides_win_and_none_ignored():
    cfg = config_from_dict({"experiment_name": "a", "epochs": 3}, {"epochs": 5, "batch": None})
    assert cfg.epochs == 5 and cfg.batch == 8


@pytest.mark.parametrize(
    "bad",
    [
        {"epochs": 0}, {"batch": -1}, {"imgsz": 650}, {"optimizer": "Lion"}, {"lr0": 0},
        {"fraction": 1.5}, {"model": "yolov8n"}, {"experiment_name": "../escape"},
        {"experiment_name": "has space"}, {"workers": -1}, {"epochs": True},
        {"train_args": {"data": "other.yaml"}}, {"train_args": {"epochs": 99}},
    ],
)
def test_invalid_config_rejected(bad):
    with pytest.raises(DetectorConfigError):
        config_from_dict({"experiment_name": "ok", **bad})


def test_unknown_or_missing_keys_rejected():
    with pytest.raises(DetectorConfigError, match="Unknown"):
        config_from_dict({"experiment_name": "a", "epoch": 3})
    with pytest.raises(DetectorConfigError, match="experiment_name"):
        config_from_dict({"epochs": 3})


def test_ultralytics_args_are_deterministic_and_managed(tmp_path):
    cfg = _cfg(seed=7, train_args={"fliplr": 0.0})
    a = ultralytics_train_args(cfg, tmp_path / "d.yaml", tmp_path / "runs" / "t")
    assert a == ultralytics_train_args(cfg, tmp_path / "d.yaml", tmp_path / "runs" / "t")
    assert a["seed"] == 7 and a["deterministic"] is True and a["fliplr"] == 0.0
    assert a["project"] == str(tmp_path / "runs") and a["name"] == "t"
    assert Path(a["project"]).is_absolute()


# --- dataset ----------------------------------------------------------------


def test_resolve_dataset_uses_yaml_dir_not_cwd(dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # 'path: .' must resolve to the yaml's folder
    ds = resolve_dataset(dataset / "data.yaml")
    assert ds.root == dataset.resolve()
    assert set(ds.splits) == {"train", "val", "test"}
    resolved = yaml.safe_load(write_resolved_yaml(ds, tmp_path).read_text(encoding="utf-8"))
    assert Path(resolved["path"]).is_absolute() and resolved["nc"] == 6
    assert resolved["names"] == DETECTOR_CLASSES


def test_detector_class_mapping_is_locked():
    assert DETECTOR_CLASSES == {0: "name", 1: "date_of_birth", 2: "gender", 3: "aadhaar_number",
                                4: "pan_number", 5: "fathers_name"}


def _rewrite(dataset, **changes):
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    data.update(changes)
    (dataset / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"names": {0: "name", 1: "date_of_birth"}}, "vocabulary"),          # wrong class count
        ({"names": {**DETECTOR_CLASSES, 6: "class_4"}}, "vocabulary"),       # extra class
        ({"names": {**DETECTOR_CLASSES, 0: "full_name"}}, "vocabulary"),     # renamed class
        ({"nc": 7}, "nc=7"),
        ({"val": "missing/images"}, "not found"),
        ({"path": "does_not_exist"}, "root does not exist"),
    ],
)
def test_invalid_dataset_rejected(dataset, changes, match):
    _rewrite(dataset, **changes)
    with pytest.raises(DetectorDatasetError, match=match):
        resolve_dataset(dataset / "data.yaml")


def test_missing_required_split_and_descriptor(dataset, tmp_path):
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    del data["val"]
    (dataset / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(DetectorDatasetError, match="'val'"):
        resolve_dataset(dataset / "data.yaml")
    with pytest.raises(DetectorDatasetError, match="not found"):
        resolve_dataset(tmp_path / "nope.yaml")


def test_empty_split_rejected(dataset):
    (dataset / "valid" / "images" / "a.jpg").unlink()
    with pytest.raises(DetectorDatasetError, match="no images"):
        resolve_dataset(dataset / "data.yaml")


# --- run directory ----------------------------------------------------------


def test_prepare_run_dir_creates_and_never_overwrites(tmp_path):
    cfg = _cfg(experiment_name="smoke", output_dir=Path("experiments/runs"))
    run_dir = prepare_run_dir(cfg, tmp_path)
    assert run_dir == (tmp_path / "experiments" / "runs" / "smoke").resolve() and run_dir.is_dir()
    with pytest.raises(RunExistsError):
        prepare_run_dir(cfg, tmp_path)


# --- metrics ----------------------------------------------------------------


def test_extract_detection_metrics_maps_classes():
    box = SimpleNamespace(
        mp=0.5, mr=0.4, map50=0.3, map=0.2,
        ap_class_index=np.array([0, 3]),  # only two classes had targets
        p=np.array([0.9, 0.1]), r=np.array([0.8, 0.2]), ap50=np.array([0.7, 0.3]), ap=np.array([0.6, 0.25]),
    )
    m = extract_detection_metrics(SimpleNamespace(box=box), DETECTOR_CLASSES)
    assert m["overall"] == {"precision": 0.5, "recall": 0.4, "mAP50": 0.3, "mAP50-95": 0.2}
    assert set(m["per_class"]) == set(DETECTOR_CLASSES.values())
    assert m["per_class"]["aadhaar_number"] == {"class_id": 3, "precision": 0.1, "recall": 0.2,
                                                "mAP50": 0.3, "mAP50-95": 0.25}
    assert m["per_class"]["pan_number"]["mAP50"] is None


def test_cache_patch_keeps_version_stamp_and_skips_write(tmp_path):
    """Regression: Ultralytics' get_labels pops cache['version'] set by the save function."""
    pytest.importorskip("ultralytics")
    import ultralytics.data.dataset as ul_dataset

    from idocr.training.detection.train import no_dataset_cache_writes

    original = ul_dataset.save_dataset_cache_file
    cache, path = {"labels": []}, tmp_path / "labels.cache"
    with no_dataset_cache_writes():
        ul_dataset.save_dataset_cache_file("p", path, cache, "1.0.3")
    assert cache["version"] == "1.0.3"
    assert not path.exists()
    assert ul_dataset.save_dataset_cache_file is original


def test_environment_record_is_yaml_safe():
    """Regression: torch.__version__ is a str subclass that yaml.safe_dump rejects."""
    from idocr.training.detection.train import _environment

    env = _environment()
    assert yaml.safe_load(yaml.safe_dump(env)) == env
    assert all(v is None or type(v) in (str, bool) for v in env.values())


# --- end-to-end with mocked Ultralytics ----------------------------------------


class _FakeYOLO:
    calls: list = []

    def __init__(self, model):
        self.model = model

    def train(self, **kwargs):
        _FakeYOLO.calls.append(("train", self.model, kwargs))
        weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"x")
        (weights / "last.pt").write_bytes(b"x")

    def val(self, **kwargs):
        _FakeYOLO.calls.append(("val", self.model, kwargs))
        n = len(DETECTOR_CLASSES)
        box = SimpleNamespace(mp=0.1, mr=0.1, map50=0.1, map=0.05, ap_class_index=np.arange(n),
                              p=np.zeros(n), r=np.zeros(n), ap50=np.zeros(n), ap=np.zeros(n))
        return SimpleNamespace(box=box)


@pytest.fixture
def fake_ultralytics(monkeypatch):
    _FakeYOLO.calls = []
    pkg = types.ModuleType("ultralytics")
    pkg.YOLO = _FakeYOLO
    data_pkg = types.ModuleType("ultralytics.data")
    ds_mod = types.ModuleType("ultralytics.data.dataset")
    sentinel = object()
    ds_mod.save_dataset_cache_file = sentinel
    pkg.data = data_pkg
    data_pkg.dataset = ds_mod
    for name, mod in {"ultralytics": pkg, "ultralytics.data": data_pkg, "ultralytics.data.dataset": ds_mod}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delenv("YOLO_OFFLINE", raising=False)
    return ds_mod, sentinel


def test_train_pipeline_with_mocked_framework(dataset, tmp_path, fake_ultralytics, monkeypatch):
    ds_mod, sentinel = fake_ultralytics
    cfg = _cfg(experiment_name="smoke", dataset=dataset / "data.yaml", output_dir=tmp_path / "runs",
               validate_dataset=False)
    before = sorted(p.relative_to(dataset).as_posix() for p in dataset.rglob("*"))

    record = train(cfg, root=tmp_path)

    run_dir = tmp_path / "runs" / "smoke"
    assert record["status"] == "completed"
    assert set(record["checkpoints"]) == {"best.pt", "last.pt"}
    assert (run_dir / "metrics.json").is_file() and (run_dir / "data.resolved.yaml").is_file()
    saved = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    assert saved["status"] == "completed" and saved["config"]["seed"] == 42
    assert saved["dataset"]["classes"] == DETECTOR_CLASSES

    (kind, model, targs), (vkind, vmodel, vargs) = _FakeYOLO.calls
    assert (kind, model) == ("train", "yolov8n.yaml") and targs["data"].endswith("data.resolved.yaml")
    assert vkind == "val" and vmodel.endswith("best.pt") and vargs["split"] == "val"
    assert ds_mod.save_dataset_cache_file is sentinel  # cache patch restored
    import os
    assert os.environ.get("YOLO_OFFLINE") == "1"
    assert sorted(p.relative_to(dataset).as_posix() for p in dataset.rglob("*")) == before  # dataset untouched

    with pytest.raises(RunExistsError):
        train(cfg, root=tmp_path)


def test_failed_training_is_recorded(dataset, tmp_path, fake_ultralytics, monkeypatch):
    def boom(self, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(_FakeYOLO, "train", boom)
    cfg = _cfg(experiment_name="bad", dataset=dataset / "data.yaml", output_dir=tmp_path / "runs",
               validate_dataset=False)
    with pytest.raises(RuntimeError):
        train(cfg, root=tmp_path)
    saved = yaml.safe_load((tmp_path / "runs" / "bad" / "run_config.yaml").read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and "simulated failure" in saved["error"]
