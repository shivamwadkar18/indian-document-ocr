# indian-document-ocr

OCR and document understanding for Indian identity documents: take a document
image, produce structured, validated fields.

## Current Project Status

- **Day 1 & Day 2 Completed**: Foundation, dataset audit, visualization, manual class mappings, and leakage-free dataset splitting completed.
- **Processed Datasets**: Clean, leakage-free `train`, `valid`, and `test` splits created under `data/processed/aadhaar/` and `data/processed/pan/`.
- **Source-Level Isolation**: 0 cross-split source groups, 0 cross-split duplicate images.
- **Class Mappings**: Identified and configured in [`configs/class_mapping.yaml`](file:///c:/Projects/indian-document-ocr/configs/class_mapping.yaml).
- **Tests**: 77 tests passing (`pytest`).
- **Raw Data**: Verified immutable and untouched in `data/raw/` (`RAW DATA MODIFIED: NO`).
- **Aadhaar Class 4**: Unresolved (`UNKNOWN` / `null`, preserved without modification).

For documentation, see:
- [docs/DAY_01.md](file:///c:/Projects/indian-document-ocr/docs/DAY_01.md) — Day 1: Scaffolding, Audits, Visualization & Class Identification
- [docs/DAY_02.md](file:///c:/Projects/indian-document-ocr/docs/DAY_02.md) — Day 2: Source Grouping, Deduplication & Leakage-Free Splitting
- [data/reports/leakage_report.md](file:///c:/Projects/indian-document-ocr/data/reports/leakage_report.md) — Full Leakage & Split Audit Report

## 1. Project overview

The system is a modular ML pipeline. Each stage sits behind a small interface
so that models can be chosen, trained and swapped independently.

## 2. Goals

- Classify the document type of an input image.
- Detect and recognize the text on it.
- Extract document-specific fields (name, DOB, ID number, …) into structured output.
- Validate extracted fields.
- Expose `image -> structured JSON` to a future backend.

## 3. Supported documents (initial)

| Document        | Training data source                     | Processed Status |
|-----------------|------------------------------------------|------------------|
| Aadhaar         | Real dataset (YOLO annotations)          | Processed & Leakage-Free (2,646 images) |
| PAN             | Real dataset (YOLO annotations)          | Processed & Leakage-Free (1,726 images) |
| Driving Licence | Synthetic, fictional data (to be built)   | Pending generation |

## 4. ML architecture

```
IMAGE
  -> preprocessing              idocr.data.preprocessing.Preprocessor
  -> document classification    idocr.models.DocumentClassifier     -> ClassificationResult
  -> text detection             idocr.models.TextDetector           -> [TextRegion{bbox, confidence}]
  -> OCR / text recognition     idocr.models.TextRecognizer         -> RecognitionResult{text, confidence}
  -> field extraction           idocr.extraction.FieldExtractor     -> {name: ExtractedField}
  -> validation                 idocr.validation.Validator          -> ValidationResult
  -> structured result          idocr.types.PipelineResult.to_dict()
```

`idocr.pipeline.DocumentPipeline` wires the stages together through dependency
injection. Conventions:

- Images are numpy arrays, `H x W x 3`, `uint8`, RGB (`idocr.types.ImageArray`).
- Interfaces do not depend on PyTorch; only future model implementations will.
- Pipeline output types (`idocr/types.py`) are separate from the training
  annotation schema (`idocr/data/annotation_schema.py`, **draft**).

Architectures for detection, recognition and classification are **not chosen yet**.

## 5. Repository structure

```
configs/            YAML configs (default.yaml, synthetic_driving_license.yaml)
data/
  raw/{aadhaar,pan,driving_license}/   original datasets (local only, git-ignored)
  processed/        converted datasets (git-ignored)
  synthetic/        generated synthetic data (git-ignored)
  reports/          audit reports (git-ignored — may contain sample annotations)
docs/               design notes and open decisions
experiments/        training run outputs (git-ignored)
models/             model weights / checkpoints (git-ignored)
notebooks/          exploration notebooks
scripts/            CLI entry points (discover_datasets.py, audit_dataset.py)
src/idocr/
  types.py          pipeline data types (BBox, TextRegion, PipelineResult, ...)
  data/
    discovery.py    locate raw datasets
    annotation_schema.py  unified annotation representation (DRAFT)
    audit/          read-only dataset audit
    converters/     source format -> unified schema (per format, after audit)
    preprocessing/  image preprocessing interface
    augmentation/   training augmentation interface
    synthetic/driving_license/  synthetic DL generator scaffold
  models/{classifier,detector,recognizer}/   model interfaces
  training/{detection,recognition,classification}/  training entry points (placeholders)
  extraction/       aadhaar.py, pan.py, driving_license.py (placeholders)
  validation/       validator interface
  evaluation/       detection.py, recognition.py, end_to_end.py
  pipeline/         document_pipeline.py
  utils/            paths, config, image I/O, logging
tests/
```

The package lives at `src/idocr/` (a named package under `src/`) so imports are
`import idocr...` rather than `import src...`, and so `data`/`models`/`utils`
don't collide with other top-level names.

## Setup

Python 3.11.

```bash
py -3.11 -m venv .venv            # Windows; use python3.11 elsewhere
.venv\Scripts\activate            # source .venv/bin/activate on Linux/macOS
pip install -e ".[dev,analysis]"  # or: pip install -r requirements.txt && pip install -e .
pytest
```

PyTorch is **not** installed by default because the correct wheel depends on
your GPU/CUDA setup. Install it when training work begins, using
https://pytorch.org/get-started/locally/.

## 6. Dataset placement

Copy datasets in **as-is** (do not rename or restructure them):

```
data/raw/aadhaar/<dataset contents>
data/raw/pan/<dataset contents>
data/raw/driving_license/   (reserved; DL data will be synthetic)
```

Check placement (run scripts with the project venv's Python — the package is
installed there, so the system `python` raises `ModuleNotFoundError`):

```bash
python scripts/discover_datasets.py
```

## 7. Dataset audit workflow

```bash
python scripts/audit_dataset.py --document aadhaar
python scripts/audit_dataset.py --document pan
```

Outputs in `data/reports/` (git-ignored): `<document>_report.json` per dataset
and `dataset_summary.md`, which is regenerated from all reports on every run.

The audit is **read-only**. It opens files for reading only, refuses to write
reports inside the dataset, and hashes every source file before and after the
run. If anything changed, it fails with `SOURCE DATA MODIFIED`. It covers:

- split layout (`<split>/images` + `<split>/labels`), unexpected files and directories
- every image: corruption, format, channels, EXIF orientation, dimensions, aspect ratios
- image/label pairing, verified per split (identical filename stem), plus
  case-collision checks
- label format, detected by registered `LabelInspector`s. `YoloTxtInspector` is
  verified on Aadhaar and PAN. It reports class IDs, counts, instances per image,
  geometry per class, and malformed, out-of-range, zero-size, duplicate or tiny annotations.
- exact and near-duplicate images, label consistency between duplicates, and
  cross-split leakage through shared Roboflow source names
- **redacted** label previews: only class IDs and coordinates are kept, and
  all other tokens are masked

Class *names* are not present in either dataset. Reports say
"Meaning not determinable from annotation files alone."

### Annotation visualization and class identification (local only)

```powershell
.venv\Scripts\python scripts\visualize_annotations.py --dataset aadhaar --split all --samples-per-class 5
.venv\Scripts\python scripts\visualize_annotations.py --dataset pan --split train --samples-per-class 5 --seed 7
```

Options: `--split train|valid|test|all`, `--samples-per-class N`, `--mixed-samples N`,
`--anomaly-samples N`, `--output-dir PATH`, `--seed N`, `--tile-size PX`, `--columns N`.

Outputs go to `data/reports/visualizations/<dataset>/<split>/` (git-ignored):

- `class_<id>.jpg`: the per-class contact sheet, with that class drawn thicker
- `mixed.jpg`: a representative mix of images
- `anomaly_<kind>.jpg`: anomalies found in the data, such as an empty label file,
  missing core classes, duplicate class instances, polygons, rare classes, or
  unusual annotation counts
- `index.json`: tile metadata (filename, class IDs, annotation type, dimensions)
- `../class_identification_report.md`: rebuilt from every `index.json`, with a
  blank *My interpretation* column

How it behaves:

- **Sampling is deterministic.** The seed fixes which images are chosen. Samples
  are spread across Roboflow source images, so augmented copies don't fill a sheet.
- **Images are only shrunk, never stretched.** They are downscaled with their
  aspect ratio preserved.
- **Labels show only `class=<id>`.** Class meanings are intentionally not
  inferred, and there is no OCR.
- **Raw datasets are read-only.** The tool refuses to write under `data/raw/`.
- **Everything stays on the local machine.** Nothing is uploaded and no external
  service is called.

After you inspect the sheets, record each class's meaning in
[`configs/class_mapping.yaml`](configs/class_mapping.yaml). Set `name` (a short
snake_case field id) and `description` per class ID, and leave a value `null` if
it's unclear. Class IDs are per dataset: Aadhaar `0` and PAN `0` are unrelated.

## 8. Future training workflow

1. Convert audited sources into the unified schema (`idocr.data.converters`) -> `data/processed/`.
2. Generate synthetic Driving Licence data -> `data/synthetic/`.
3. Train the detector, recognizer and classifier (`idocr.training.*`). Write checkpoints
   to `models/` and run outputs to `experiments/`.
4. Evaluate with `idocr.evaluation` (IoU/precision/recall/mAP, CER/WER/exact match,
   field- and document-level accuracy).

## 9. Future inference workflow

```python
from idocr.pipeline import DocumentPipeline
pipeline = DocumentPipeline(classifier=..., detector=..., recognizer=...,
                            extractors=default_extractors(), validators=...)
result = pipeline.run_file("card.jpg")
payload = result.to_dict()   # JSON-ready structured output
```

A backend will later wrap exactly this call. No HTTP API exists yet.

## 10. Team and Agent Responsibilities

| Role / Agent | Responsibilities |
| :--- | :--- |
| **Project Owner (User)** | ML architecture, dataset & model decisions, requirements, manual annotation class verification |
| **Claude Code (Previous Agent)** | Initial project scaffolding, dataset audit tooling, local visualization tooling, test suite setup |
| **Antigravity (Current Agent)** | Ongoing development, Day 1 documentation, leakage-free data processing, model training, pipeline integration |
| **Frontend Developer** | Separate frontend application |

## Security and data handling

Aadhaar and PAN images are sensitive personal data.

- Raw, processed and synthetic data, audit reports, weights and experiment outputs
  are all git-ignored. **Never commit identity documents.**
- Datasets stay on the local machine. Do not upload them or send their contents
  to external services.
- No public demo may contain real identity documents.
- Synthetic Driving Licence data must use fictional identities and identifiers only.
