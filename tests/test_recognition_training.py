import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from idocr.data.recognition.dataset import RecognitionDatasetConfig, generate_recognition_dataset  # noqa: E402
from idocr.data.recognition.generator import SUPPORTED_FIELDS, resolve_font_path  # noqa: E402
from idocr.models.recognizer import Vocabulary  # noqa: E402
from idocr.models.recognizer.crnn import CRNNRecognizer, load_checkpoint  # noqa: E402
from idocr.training.detection.train import RunExistsError  # noqa: E402
from idocr.training.recognition import (  # noqa: E402
    RecognizerConfigError,
    config_from_dict,
    load_recognizer_config,
    recognition_metrics,
    train,
)
from idocr.training.recognition.data import RecognitionDataset, collate_batch, read_split  # noqa: E402
from idocr.training.recognition.train import resolve_device  # noqa: E402
from idocr.utils.paths import find_project_root  # noqa: E402

ROOT = find_project_root()


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory) -> Path:
    try:
        font = str(resolve_font_path())
    except FileNotFoundError:
        pytest.skip("no OCR font installed")
    config = RecognitionDatasetConfig.from_yaml(ROOT / "configs" / "recognition" / "recognition_dataset.yaml")
    config = dataclasses.replace(
        config, seed=5, total_samples=120,
        generator=dataclasses.replace(config.generator, font_path=font),
    )
    out = tmp_path_factory.mktemp("recognition")
    generate_recognition_dataset(config, out)
    return out


# --- config ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["crnn_smoke_cpu.yaml", "crnn_smoke_gpu.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_recognizer_config(ROOT / "configs" / "recognition" / name)
    assert cfg.dataset_dir == Path("data/processed/recognition")


def test_config_validation() -> None:
    with pytest.raises(RecognizerConfigError, match="Unknown"):
        config_from_dict({"experiment_name": "x", "epochz": 1})
    with pytest.raises(RecognizerConfigError, match="experiment_name is required"):
        config_from_dict({})
    with pytest.raises(RecognizerConfigError):
        config_from_dict({"experiment_name": "x", "device": "gpu"})
    with pytest.raises(RecognizerConfigError):
        config_from_dict({"experiment_name": "x", "batch_size": 0})
    cfg = config_from_dict({"experiment_name": "x"}, overrides={"epochs": 3, "lr": None})
    assert cfg.epochs == 3 and cfg.lr == 1e-3


def test_resolve_device() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        with pytest.raises(ValueError):
            resolve_device("cuda")


# --- dataset + collate -----------------------------------------------------


def test_read_split_limit(dataset_dir: Path) -> None:
    assert len(read_split(dataset_dir / "train")) == 96
    assert len(read_split(dataset_dir / "train", limit=12)) == 12


def test_dataset_items(dataset_dir: Path) -> None:
    vocab = Vocabulary()
    ds = RecognitionDataset(dataset_dir / "valid", vocab, image_height=56)
    assert len(ds) == 12
    image, target, sample = ds[0]
    assert image.mode == "L" and image.height == 56
    assert vocab.decode(target) == sample.text
    assert sample.split == "valid"


def test_dataset_rejects_wrong_height(dataset_dir: Path) -> None:
    ds = RecognitionDataset(dataset_dir / "valid", Vocabulary(), image_height=64)
    with pytest.raises(ValueError):
        ds[0]


def test_collate_batch(dataset_dir: Path) -> None:
    vocab = Vocabulary()
    ds = RecognitionDataset(dataset_dir / "train", vocab, image_height=56)
    items = [ds[i] for i in range(6)]
    batch = collate_batch(items)

    max_w = max(im.width for im, _, _ in items)
    assert batch["images"].shape[:3] == (6, 1, 56)
    assert batch["images"].shape[3] >= max_w and batch["images"].shape[3] % 4 == 0
    assert batch["widths"].tolist() == [im.width for im, _, _ in items]
    assert batch["target_lengths"].tolist() == [len(t) for _, t, _ in items]
    assert batch["targets"].numel() == sum(len(t) for _, t, _ in items)
    assert set(batch["fields"]) == set(SUPPORTED_FIELDS)
    assert batch["images"].min() >= -1 and batch["images"].max() <= 1


# --- metrics ---------------------------------------------------------------


def test_recognition_metrics() -> None:
    m = recognition_metrics(["MALE", "ABCPX", "12/01/1990"], ["MALE", "ABCPJ", "12/01/1990"],
                            ["gender", "pan_number", "date_of_birth"])
    assert m["exact_match"] == pytest.approx(2 / 3)
    assert m["cer"] == pytest.approx(1 / 19)
    assert m["per_field"]["pan_number"] == {"cer": 0.2, "exact_match": 0.0, "count": 1}
    assert m["per_field"]["gender"]["exact_match"] == 1.0


# --- end to end ------------------------------------------------------------


def test_train_end_to_end(dataset_dir: Path, tmp_path: Path) -> None:
    cfg = config_from_dict({
        "experiment_name": "tiny", "dataset_dir": str(dataset_dir), "output_dir": str(tmp_path),
        "epochs": 2, "batch_size": 16, "device": "cpu", "hidden_size": 32, "lstm_layers": 1,
        "max_train_samples": 48, "log_every": 100,
    })
    record = train(cfg, root=ROOT)
    run_dir = tmp_path / "tiny"

    assert record["status"] == "completed"
    assert record["dataset"]["train_samples"] == 48
    assert record["dataset"]["valid_samples"] == 12
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert [h["epoch"] for h in metrics["history"]] == [1, 2]
    assert 0 <= metrics["history"][-1]["valid"]["cer"]
    assert set(metrics["history"][-1]["valid"]["per_field"]) == set(SUPPORTED_FIELDS)

    last = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    assert last["epoch"] == 2 and "optimizer_state" in last
    assert last["preprocessing"] == {"target_height": 48, "horizontal_padding": 8, "vertical_padding": 4}

    recognizer = CRNNRecognizer.from_checkpoint(run_dir / "checkpoints" / "best.pt")
    sample = read_split(dataset_dir / "test", 1)[0]
    crop = np.asarray(Image.open(dataset_dir / "test" / sample.image_path))
    result = recognizer.recognize(crop)
    assert 0 <= result.confidence <= 1

    with pytest.raises(RunExistsError):
        train(cfg, root=ROOT)


def test_train_is_deterministic(dataset_dir: Path, tmp_path: Path) -> None:
    base = {"dataset_dir": str(dataset_dir), "output_dir": str(tmp_path), "epochs": 1, "batch_size": 16,
            "device": "cpu", "hidden_size": 32, "lstm_layers": 1, "max_train_samples": 32,
            "max_valid_samples": 6}
    a = train(config_from_dict({**base, "experiment_name": "a"}), root=ROOT)
    b = train(config_from_dict({**base, "experiment_name": "b"}), root=ROOT)
    assert a["final_valid"] == b["final_valid"]


def test_amp_requires_cuda(dataset_dir: Path, tmp_path: Path) -> None:
    cfg = config_from_dict({"experiment_name": "amp", "dataset_dir": str(dataset_dir),
                            "output_dir": str(tmp_path), "device": "cpu", "amp": True})
    with pytest.raises(ValueError):
        train(cfg, root=ROOT)


def test_train_with_augmentation(dataset_dir: Path, tmp_path: Path) -> None:
    cfg = config_from_dict({
        "experiment_name": "aug", "dataset_dir": str(dataset_dir), "output_dir": str(tmp_path),
        "epochs": 2, "batch_size": 12, "device": "cpu", "hidden_size": 32, "lstm_layers": 1,
        "max_train_samples": 24, "max_valid_samples": 6,
        "augmentation": {"enabled": True},
    })
    record = train(cfg, root=ROOT)
    assert record["status"] == "completed"
    assert record["config"]["augmentation"] == {"enabled": True}
