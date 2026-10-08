"""CRNN + CTC recognizer training on the synthetic recognition dataset.

Pipeline: validate config -> read dataset_info.json (image height,
preprocessing) -> create a new run dir (never overwrites) -> train with CTC
loss / AdamW / grad clipping -> evaluate on ``valid/`` each epoch (loss, CER,
exact match, per-field exact match) -> save ``checkpoints/last.pt`` and
``checkpoints/best.pt`` (lowest valid CER) -> write ``metrics.json``.

Run layout (``<output_dir>/<experiment_name>/``)::

    run_config.yaml        resolved config, environment, dataset, status
    checkpoints/last.pt    weights + optimizer + vocab + preprocessing
    checkpoints/best.pt
    metrics.json           per-epoch history + best epoch
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from idocr.data.augmentation.ocr import OcrLineAugmenter
from idocr.data.recognition.dataset import INFO_FILENAME
from idocr.evaluation.recognition import character_error_rate, exact_match_accuracy
from idocr.models.recognizer.crnn import (
    CRNN,
    CRNNConfig,
    greedy_decode,
    save_checkpoint,
)
from idocr.models.recognizer.vocab import BLANK_INDEX, Vocabulary
from idocr.training.detection.train import _environment, _write_run_config, prepare_run_dir
from idocr.training.recognition.config import RecognizerTrainConfig, config_from_dict
from idocr.training.recognition.data import RecognitionDataset, collate_batch
from idocr.utils.paths import find_project_root

METRICS_FILENAME = "metrics.json"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"device {device!r} requested but CUDA is not available")
    return torch.device(device)


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def read_preprocessing(dataset_dir: Path) -> dict[str, int]:
    """Preprocessing used to build the dataset (from dataset_info.json)."""
    info_path = dataset_dir / INFO_FILENAME
    if not info_path.is_file():
        raise FileNotFoundError(f"missing {info_path}; generate the recognition dataset first")
    gen = json.loads(info_path.read_text(encoding="utf-8"))["generator_config"]
    return {k: int(gen[k]) for k in ("target_height", "horizontal_padding", "vertical_padding")}


def build_loaders(
    cfg: RecognizerTrainConfig,
    dataset_dir: Path,
    vocab: Vocabulary,
    image_height: int,
    generator: torch.Generator,
    pin_memory: bool = False,
    preprocessing: Mapping[str, int] | None = None,
) -> tuple[DataLoader, DataLoader]:
    aug_cfg = cfg.augment_config
    augment = OcrLineAugmenter(aug_cfg, **(preprocessing or {})) if aug_cfg.enabled else None
    train_ds = RecognitionDataset(dataset_dir / "train", vocab, image_height=image_height,
                                  limit=cfg.max_train_samples, augment=augment, seed=cfg.seed)
    valid_ds = RecognitionDataset(dataset_dir / "valid", vocab, image_height=image_height,
                                  limit=cfg.max_valid_samples)
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_batch,
                  worker_init_fn=_seed_worker, pin_memory=pin_memory,
                  persistent_workers=cfg.num_workers > 0)
    # Workers must be re-created each epoch to see train_ds.set_epoch().
    train_persistent = cfg.num_workers > 0 and augment is None
    return (
        DataLoader(train_ds, shuffle=True, generator=generator,
                   **{**common, "persistent_workers": train_persistent}),
        DataLoader(valid_ds, shuffle=False, **common),
    )


def warmup_cosine(total_steps: int, warmup_fraction: float = 0.05):
    """LR multiplier: linear warmup, then cosine decay to 0 over ``total_steps``."""
    warmup = max(1, int(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def ctc_loss_fn() -> nn.CTCLoss:
    return nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)


def compute_loss(model: CRNN, batch: Mapping[str, Any], criterion: nn.CTCLoss,
                 device: torch.device, amp: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward + CTC loss. Returns ``(loss, log_probs)``."""
    images = batch["images"].to(device, non_blocking=True)
    widths = batch["widths"].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        log_probs = model(images, widths)  # float32 (T, B, C)
    loss = criterion(log_probs, batch["targets"].to(device), CRNN.output_lengths(widths),
                     batch["target_lengths"].to(device))
    return loss, log_probs


def recognition_metrics(predictions: list[str], references: list[str], fields: list[str]) -> dict[str, Any]:
    by_field: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for p, r, f in zip(predictions, references, fields):
        by_field[f].append((p, r))
    return {
        "cer": character_error_rate(predictions, references),
        "exact_match": exact_match_accuracy(predictions, references),
        "per_field": {
            f: {
                "cer": character_error_rate([p for p, _ in pairs], [r for _, r in pairs]),
                "exact_match": exact_match_accuracy([p for p, _ in pairs], [r for _, r in pairs]),
                "count": len(pairs),
            }
            for f, pairs in sorted(by_field.items())
        },
    }


@torch.inference_mode()
def evaluate(model: CRNN, loader: DataLoader, vocab: Vocabulary, device: torch.device,
             amp: bool = False) -> dict[str, Any]:
    model.eval()
    criterion = ctc_loss_fn()
    total_loss, n = 0.0, 0
    predictions: list[str] = []
    references: list[str] = []
    fields: list[str] = []
    for batch in loader:
        loss, log_probs = compute_loss(model, batch, criterion, device, amp)
        total_loss += loss.item() * len(batch["texts"])
        n += len(batch["texts"])
        decoded = greedy_decode(log_probs, CRNN.output_lengths(batch["widths"]), vocab)
        predictions += [text for text, _ in decoded]
        references += batch["texts"]
        fields += batch["fields"]
    return {"loss": total_loss / max(n, 1), **recognition_metrics(predictions, references, fields),
            "examples": [{"pred": p, "ref": r} for p, r in list(zip(predictions, references))[:6]]}


def train_one_epoch(model: CRNN, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device, cfg: RecognizerTrainConfig,
                    epoch: int, scheduler: torch.optim.lr_scheduler.LRScheduler | None = None) -> float:
    model.train()
    criterion = ctc_loss_fn()
    total, n = 0.0, 0
    for step, batch in enumerate(loader, 1):
        loss, _ = compute_loss(model, batch, criterion, device, cfg.amp)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        # Advance scheduler only when optimizer.step() was actually executed
        if scheduler is not None and scale_before <= scale_after:
            scheduler.step()
        total += loss.item() * len(batch["texts"])
        n += len(batch["texts"])
        if step % cfg.log_every == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} loss {total / n:.4f}", flush=True)
    return total / max(n, 1)


def train(config: RecognizerTrainConfig | Mapping[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    """Train the CRNN recognizer. Returns the final run record."""
    cfg = config if isinstance(config, RecognizerTrainConfig) else config_from_dict(config)
    root_path = Path(root).resolve() if root is not None else find_project_root()
    dataset_dir = cfg.dataset_dir if cfg.dataset_dir.is_absolute() else root_path / cfg.dataset_dir
    preprocessing = read_preprocessing(dataset_dir)
    image_height = preprocessing["target_height"] + 2 * preprocessing["vertical_padding"]

    device = resolve_device(cfg.device)
    if cfg.amp and device.type != "cuda":
        raise ValueError("amp requires a CUDA device")
    generator = seed_everything(cfg.seed)
    vocab = Vocabulary()
    train_loader, valid_loader = build_loaders(cfg, dataset_dir, vocab, image_height, generator,
                                               pin_memory=device.type == "cuda",
                                               preprocessing=preprocessing)

    model = CRNN(CRNNConfig(num_classes=vocab.num_classes, image_height=image_height,
                            hidden_size=cfg.hidden_size, lstm_layers=cfg.lstm_layers,
                            dropout=cfg.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(steps))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp)

    run_dir = prepare_run_dir(cfg, root_path)
    record: dict[str, Any] = {
        "experiment_name": cfg.experiment_name,
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "device": str(device),
        "dataset": {"dir": dataset_dir.as_posix(), "train_samples": len(train_loader.dataset),
                    "valid_samples": len(valid_loader.dataset), "preprocessing": preprocessing},
        "model": {"architecture": "CRNN (CNN -> BiLSTM -> Linear, CTC)",
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "alphabet": vocab.alphabet},
        "environment": _environment(),
    }
    _write_run_config(run_dir, record)

    history: list[dict[str, Any]] = []
    best_cer = float("inf")
    best_epoch = None
    try:
        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            train_loader.dataset.set_epoch(epoch)
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch,
                                         scheduler)
            val = evaluate(model, valid_loader, vocab, device, cfg.amp)
            entry = {"epoch": epoch, "train_loss": train_loss, "valid": val,
                     "seconds": round(time.perf_counter() - started, 1)}
            history.append(entry)
            print(f"epoch {epoch}: train_loss {train_loss:.4f} valid_loss {val['loss']:.4f} "
                  f"cer {val['cer']:.4f} exact {val['exact_match']:.4f} ({entry['seconds']}s)", flush=True)

            meta = dict(preprocessing=preprocessing, epoch=epoch, valid_cer=val["cer"],
                        valid_exact_match=val["exact_match"], train_config=cfg.to_dict())
            save_checkpoint(run_dir / "checkpoints" / "last.pt", model, vocab,
                            optimizer_state=optimizer.state_dict(), **meta)
            if val["cer"] < best_cer:
                best_cer, best_epoch = val["cer"], epoch
                save_checkpoint(run_dir / "checkpoints" / "best.pt", model, vocab, **meta)
            (run_dir / METRICS_FILENAME).write_text(
                json.dumps({"best_epoch": best_epoch, "best_valid_cer": best_cer, "history": history},
                           indent=2), encoding="utf-8")
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      finished_at=datetime.now().isoformat(timespec="seconds"))
        _write_run_config(run_dir, record)
        raise

    record.update(
        status="completed",
        finished_at=datetime.now().isoformat(timespec="seconds"),
        run_dir=run_dir.as_posix(),
        checkpoints={p.name: p.relative_to(run_dir).as_posix()
                     for p in sorted((run_dir / "checkpoints").glob("*.pt"))},
        best_epoch=best_epoch,
        best_valid_cer=best_cer,
        final_valid={k: history[-1]["valid"][k] for k in ("loss", "cer", "exact_match")},
        metrics_file=METRICS_FILENAME,
    )
    _write_run_config(run_dir, record)
    return record

