# Day 3 — ML Environment Preparation & Detector Strategy

## Overview

Day 3 establishes the machine learning environment and detector training strategy for Claude Code's model development. The runtime environment was audited, PyTorch was installed and verified, the detector dataset properties were evaluated, and the baseline framework and initial training configurations were formalized.

---

## 1. System & Runtime Environment

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 10 Pro (10.0.19045, AMD64) | 64-bit |
| **Python Version** | Python 3.11.9 | In `.venv` |
| **Pip Version** | 26.2.1 | In `.venv` |
| **CPU** | Intel(R) Core(TM) i7-7600U @ 2.80GHz | 2 physical cores / 4 logical threads |
| **RAM** | ~16 GB system memory | |
| **GPU / Video Controller** | Intel(R) HD Graphics 620 | Integrated GPU (no dedicated NVIDIA GPU) |
| **CUDA Status** | **Unavailable** (`torch.cuda.is_available() == False`) | Local environment runs on CPU |
| **PyTorch Version** | `2.14.1+cpu` | Installed from PyTorch official CPU wheels |
| **Torchvision Version** | `0.29.1+cpu` | Installed from PyTorch official CPU wheels |
| **OpenCV Version** | `5.0.0.93` | Pre-installed |
| **Ultralytics** | Not installed yet | Recommended for detector implementation |

---

## 2. Detector Dataset Status

- **Location**: `data/processed/detector/` (`data.yaml`, `train/`, `valid/`, `test/`)
- **Total Storage Size**: 379.23 MB
- **Image Count**: 4,372 total (3,060 train, 654 valid, 658 test)
- **Target Count**: 17,102 targets across 6 classes
- **Image Aspect Ratios**: Mixed dimensions (e.g. 640x640, 720x1280, 2339x1653, 1000x636) representing standard document scans and mobile captures.

---

## 3. Recommended Baseline Detector Framework

### Choice: **Ultralytics YOLO (YOLOv8 / YOLO11)**

**Rationale**:
1. **Direct Compatibility**: Works natively with the generated `data/processed/detector/` layout and `data.yaml` descriptor without intermediate conversion.
2. **Unified Metrics**: Computes per-class and overall mAP50, mAP50-95, precision, recall, confusion matrices, and PR curves automatically during validation.
3. **Execution Flexibility**: Seamless execution across CPU (`device='cpu'`) for local debugging/testing and CUDA (`device=0`) if trained on accelerated cloud hardware.
4. **Reproducibility & Experiment Tracking**: Automatically logs hyperparameter configurations (`args.yaml`), per-epoch loss metrics (`results.csv`), and best/last checkpoints (`best.pt`, `last.pt`).

---

## 4. Baseline Model Recommendation

### Initial Baseline: **YOLOv8n (nano)** (~3.2M parameters)

**Rationale**:
- **Lightweight & Fast**: Extremely compact memory footprint (~6 MB weights) and fast execution on CPU / local development.
- **Fast Feedback Loop**: Enables rapid validation of training scripts, loss convergence, metric calculation, and pipeline integration before scaling compute.
- **Scaling Path**: Once the baseline pipeline is verified, scaling to `yolov8s` (small, 11.2M params) or `yolo11s` is straightforward with zero code alterations.

---

## 5. Recommended Initial Training Configuration

| Parameter | Recommended Value | Rationale |
| :--- | :--- | :--- |
| **Image Size (`imgsz`)** | `640` | Standard resolution for field detection; preserves document text clarity. |
| **Batch Size (`batch`)** | `16` (or `8` on CPU) | Balances gradient stability and memory consumption. |
| **Epochs (`epochs`)** | `50` | Sufficient for convergence on 4.3k images with pre-trained weights. |
| **Patience (`patience`)** | `10` | Early stopping to prevent overfitting if validation mAP plateaus. |
| **Device (`device`)** | `'cpu'` (local) / `0` (GPU) | Explicit device selection. |
| **Workers (`workers`)** | `2` | Matches 2 physical CPU cores. |
| **Optimizer (`optimizer`)** | `SGD` or `AdamW` | `SGD` (momentum=0.937) for stable generalization. |
| **Initial Learning Rate (`lr0`)** | `0.01` (SGD) / `0.001` (AdamW) | Standard warm-up with cosine learning rate decay. |
| **Augmentation Strategy** | Mild affine/scale; disable horizontal flip (`fliplr=0.0`) | Text documents are horizontally directional; avoid horizontal flipping. |

---

## 6. Recommended Experiment Directory Layout

```text
experiments/
├── configs/             # YAML experiment configurations
├── runs/                # YOLO training outputs (weights, metrics, plots)
│   ├── baseline_yolov8n/
│   │   ├── weights/     # best.pt, last.pt
│   │   ├── args.yaml
│   │   ├── results.csv
│   │   └── val/         # validation predictions & confusion matrices
└── checkpoints/         # Candidate model weights for downstream OCR pipeline
```

---

## 7. Verification & Test Status

- **PyTorch / Torchvision Import**: Verified operational in `.venv` (`torch.rand(2, 2)` CPU execution passed).
- **Dependencies Audit**: 9/9 `requirements.txt` packages and all `pyproject.toml` dependencies satisfied.
- **Dependency Conflicts (`pip check`)**: **No broken requirements found**.
- **Project Imports**: All core, schema, converter, splitting, detection, and visualization modules import cleanly.
- **Detector Dataset Accessibility**: Validated read access across all 4,372 images, 4,372 labels, and `data.yaml`.
- **Full Test Suite**: **107 / 107 tests passing**.
- **Source Datasets**: Raw and processed source datasets remain 100% intact.

---

## 8. Final Environment Verification & Readiness Verdict

- **Environment**: Python 3.11.9, pip 26.2.1, PyTorch 2.14.1+cpu, Torchvision 0.29.1+cpu, Ultralytics 8.4.173 on Windows 10 (AMD64).
- **Dependencies Audit**: 10/10 `requirements.txt` packages and all `pyproject.toml` dependencies satisfied.
- **Dependency Conflicts (`pip check`)**: **No broken requirements found**.
- **Project Imports**: All core, schema, converter, splitting, detection, training, and visualization modules import cleanly.
- **Detector Dataset Accessibility**: Validated read access across all 4,372 images, 4,372 labels, and `data.yaml`.
- **Full Test Suite**: **143 / 143 tests passing** (107 existing + 36 detector training tests).
- **Source Datasets**: Raw and processed source datasets remain 100% intact (0 `.cache` files created).
- **Final Verdict**: **READY FOR CLAUDE CODE** for model training and detector experimentation.

---

## 9. Detector Training Pipeline (implemented) and Smoke Test

> **Smoke test ≠ real baseline.** Only a 1-epoch CPU smoke test was run, to
> prove the pipeline end to end. Its metrics come from a randomly initialised
> model trained for one epoch and say **nothing** about detector quality.
> The real baseline (`configs/detector/baseline_yolov8n.yaml`, 50 epochs,
> patience 10) has **not** been run.

### Framework and model

| Item | Value |
| :--- | :--- |
| Framework | Ultralytics YOLO, `ultralytics==8.4.173` (pinned; `detector` extra in `pyproject.toml`, listed in `requirements.txt`) |
| Compatibility | Installed without changing `torch 2.14.1+cpu` / `torchvision 0.29.1+cpu`; `pip check` clean |
| Model | YOLOv8n (`yolov8n.yaml`: architecture, random init — no weight download for the smoke test) |

### Implementation

| Part | Location |
| :--- | :--- |
| Config dataclass + validation | `src/idocr/training/detection/config.py` |
| Dataset checks + resolved `data.yaml` | `src/idocr/training/detection/dataset.py` |
| Metrics extraction | `src/idocr/training/detection/metrics.py` |
| Training entry point | `src/idocr/training/detection/train.py` (`train(config)`) |
| CLI | `scripts/train_detector.py --config <yaml> [--experiment-name … --epochs … --fraction …]` |
| Configs | `configs/detector/smoke_test_yolov8n.yaml`, `configs/detector/baseline_yolov8n.yaml` |
| Run outputs | `experiments/runs/<experiment_name>/` (git-ignored) |

Pipeline: validate config → check `data.yaml` (6 locked classes, splits exist) →
full `DetectorDatasetValidator` preflight → create a **new** run dir (existing runs
are never overwritten) → write `data.resolved.yaml` + `run_config.yaml` → train →
validate the best checkpoint on `valid/` → write `metrics.json`, mark the run
`completed` (or `failed` with the error).

Safeguards:

- **`data.yaml` path:** `path: .` is resolved by Ultralytics against the *working
  directory*, so the run uses a copy with an absolute dataset path. The dataset's
  own `data.yaml` is untouched.
- **No dataset writes:** Ultralytics' `labels.cache` writes next to the labels are
  suppressed during training, so `data/processed/detector/` stays byte-identical.
- **Offline:** `offline: true` sets `YOLO_OFFLINE=1`, which disables Ultralytics
  analytics and automatic downloads.
- **Plots off for the smoke test:** plots render identity-document images into the
  run dir.
- **No horizontal flips:** `fliplr: 0.0`. Seed 42, `deterministic: true`.

### Smoke test

Executed via:
```powershell
.venv\Scripts\python.exe scripts\train_detector.py --config configs\detector\smoke_test_yolov8n.yaml
```

- **Output Directory**: `experiments/runs/smoke_test_yolov8n/`
- **Execution**: 1 epoch, batch=8, imgsz=640, device=cpu, fraction=0.02 (61 train images, 654 validation images).
- **Checkpoints Created**: `weights/best.pt` (6.2 MB), `weights/last.pt` (6.2 MB).
- **Validation Executed**: Full 654-image validation run on `valid/`.
- **Status in `run_config.yaml`**: `status: completed`.
- **Dataset Integrity**: 0 `.cache` files created in `data/processed/detector/`.

### Metrics supported

`metrics.json` reports overall **precision, recall, mAP50, mAP50-95** and the same
four per class for `name`, `date_of_birth`, `gender`, `aadhaar_number`,
`pan_number` and `fathers_name`. Ultralytics' own `results.csv` holds the
per-epoch losses and metrics.

### Tests

- **Previous tests**: 107
- **New tests**: 36 (in `tests/test_detector_training.py` covering config validation, dataset resolution, directory preparation, metric extraction, and mocked training execution)
- **Final test count**: **143 passed, 0 failures**

### CPU-only limitations

This laptop has no CUDA GPU (Intel HD 620, 2 cores / 4 threads). One epoch of
YOLOv8n at 640 px with batch 8 across all 3,060 images takes about 40 minutes, so 50 epochs would take
more than a day, without counting validation. **Run the real baseline on a
CUDA GPU** (`device: "0"` in `baseline_yolov8n.yaml`). The pretrained-weights
choice (`yolov8n.pt`, which needs a one-off download, vs. random init) is still
an owner decision.
