# Day 1: Project Initialization, Auditing & Baseline Setup

## 1. Day 1 Overview

Day 1 focused on laying the foundation for the `indian-document-ocr` pipeline. This included establishing the modular codebase architecture, integrating the raw Indian identity document datasets (Aadhaar and PAN), building a read-only audit system to inspect dataset integrity and distributions, developing local visual inspection tools, manually identifying dataset field classes with the project owner, and securing immutable raw data verification alongside a 69-test validation suite.

---

## 2. Project Initialization

A modular package structure was initialized under `src/idocr/` with dedicated interfaces across the document understanding lifecycle:

* **Dataset Handling & Discovery**: Discovery, hashing, inspection, and converters (`idocr.data`).
* **Preprocessing & Augmentation**: Clean interfaces for image normalization, rotation, and augmentations (`idocr.data.preprocessing`, `idocr.data.augmentation`).
* **Synthetic Data Generation**: Scaffolding for synthetic Indian Driving Licence generation (`idocr.data.synthetic.driving_license`).
* **Model Interfaces**: Swappable abstractions for classification, text detection, and text recognition (`idocr.models`).
* **Training Entry Points**: Placeholders for isolated training pipelines (`idocr.training`).
* **Field Extraction & Validation**: Structured parsing and domain rules per document type (`idocr.extraction`, `idocr.validation`).
* **Evaluation Metrics**: Detection mAP/IoU, recognition CER/WER, and end-to-end field metrics (`idocr.evaluation`).
* **Pipeline Architecture**: Unified orchestrator connecting stages via dependency injection (`idocr.pipeline.DocumentPipeline`).
* **Testing Suite**: Automated testing suite under `tests/`.

---

## 3. Environment

* **Python Version**: Python 3.11 virtual environment (`.venv`).
* **Installed Packages**: Core development, linting, analysis, image processing, and testing dependencies installed via `pip install -e ".[dev,analysis]"`.
* **PyTorch Status**: Intentionally **not** installed during Day 1, pending finalization of model architectures, framework choices, and GPU/CUDA deployment targets.

---

## 4. Datasets

The raw identity document datasets placed into the project are:

### Summary of Raw Datasets

| Document | Directory Structure | Splits Present | Image Count | Label Count | Total Files | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Aadhaar** | `data/raw/aadhaar/{train,valid,test}/{images,labels}` | `train`, `valid`, `test` | 2,646 | 2,646 | 5,292 | Real data present |
| **PAN** | `data/raw/pan/{train,valid}/{images,labels}` | `train`, `valid` | 1,726 | 1,726 | 3,452 | Real data present (no test split) |
| **Driving Licence**| `data/raw/driving_license/` | None | 0 | 0 | 0 | Reserved for synthetic generation |

### Detailed Image Counts

* **Aadhaar**:
  * `train`: 1,852 images
  * `valid`: 529 images
  * `test`: 265 images
  * **Total**: 2,646 images
* **PAN**:
  * `train`: 1,458 images
  * `valid`: 268 images
  * `test`: 0 images (no test split exists)
  * **Total**: 1,726 images

---

## 5. Dataset Audit

Claude Code implemented and executed the read-only dataset audit infrastructure (`scripts/audit_dataset.py`, `src/idocr/data/audit/`).

The audit performed thorough checks on:
* Image and label 1:1 pairing and stem alignment.
* Image readability, formats, color channels, dimensions, and aspect ratios.
* Label syntax formatting and normalized coordinate validity `[0, 1]`.
* Class ID inventory and annotation instance distributions per image.
* Duplicate and near-duplicate detection (including Roboflow augmentation naming patterns).
* Split layout and cross-split leakage.
* Raw source file immutability via pre/post run cryptographic hashing.

### Key Audit Findings

#### Aadhaar Split Leakage
* The audit revealed severe data leakage across splits.
* The 2,646 Aadhaar images originate from approximately **239 unique source documents** that were augmented multiple times.
* Augmented versions of the same source document appear across `train`, `valid`, and `test` splits.
* **Conclusion**: The current split structure is completely unreliable for machine learning evaluation and must not be used as-is.

#### PAN Duplication and Missing Split
* The audit identified exact byte-identical duplicates, cross-split duplicates, and near-duplicate images.
* The dataset lacks a dedicated `test` split.
* **Conclusion**: PAN requires deduplication and a clean, isolated source-level split prior to model training.

---

## 6. Annotation Format

Both Aadhaar and PAN datasets use YOLO-format text annotations:

```text
<class_id> <center_x> <center_y> <width> <height>
```

with all coordinates normalized to `[0, 1]`.

### Important Characteristics
* **Geometry**: Primarily bounding boxes; Aadhaar additionally contains rare polygon annotations (class 4).
* **Field-Level Labels Only**: Annotations delimit bounding regions for specific card fields; they do **not** contain ground truth OCR text transcriptions.
* **OCR Capability**: Because ground truth text transcriptions are absent, these raw datasets cannot be used on their own for training text recognition models without additional OCR transcription data or pre-trained OCR engines.

---

## 7. Visualization

A local-only annotation visualization tool (`scripts/visualize_annotations.py`) was implemented by Claude Code:

* Renders YOLO bounding boxes and polygon contours overlaid on document images.
* Labels each annotation clearly with its numeric `class=<id>`.
* Samples representative documents across distinct source images (avoiding duplicate augmented sheets).
* Generates contact sheets per class (`class_<id>.jpg`), mixed samples (`mixed.jpg`), and anomaly contact sheets (`anomaly_<kind>.jpg`).
* Generates structured tile indices (`index.json`) and markdown identification templates (`class_identification_report.md`).
* Output artifacts reside in `data/reports/visualizations/` (git-ignored).
* **Privacy & Security**: Operates 100% locally. No document images or data are sent to external services or APIs.

---

## 8. Manually Confirmed Class Mapping

Using the local visualization contact sheets, the user inspected sample images and confirmed the following semantic meanings for the numeric class IDs:

### Aadhaar Class Mapping

| Class ID | Field Name | Description / Meaning |
| :---: | :--- | :--- |
| `0` | `aadhaar_number` | Aadhaar Number (12-digit UID) |
| `1` | `date_of_birth` | Date of Birth |
| `2` | `gender` | Gender |
| `3` | `name` | Name |
| `4` | `null` | **UNKNOWN** (Do not guess) |

### PAN Class Mapping

| Class ID | Field Name | Description / Meaning |
| :---: | :--- | :--- |
| `0` | `date_of_birth` | Date of Birth |
| `1` | `fathers_name` | Father's Name |
| `2` | `name` | Name |
| `3` | `pan_number` | PAN Number (10-character alphanumeric) |

---

## 9. Class 4 Investigation Status

Aadhaar Class 4 remains deliberately unassigned (`UNKNOWN` / `null`).

* **Audit Statistics**:
  * Total annotations: 87 instances across 86 images.
  * Geometry: 83 bounding boxes, 4 polygons.
  * Distribution: 1 image contains class 4 twice; the remaining 85 contain it once.
  * Frequency: Extremely rare relative to core fields.
* **Next Steps**: Class 4 requires further investigation into visual patterns and card variants before deciding whether to incorporate it into the unified schema, re-map it, or filter it out.

---

## 10. Configuration

The confirmed mappings have been recorded in [`configs/class_mapping.yaml`](file:///c:/Projects/indian-document-ocr/configs/class_mapping.yaml):
* Records verified `name` and `description` per class ID.
* Aadhaar class 4 is preserved as `null` / unknown without speculation.

---

## 11. Testing

The project maintains a test suite covering dataset discovery, format inspection, label validation, visualization sampling, and pipeline types.

```text
69 passed, 0 failed
```

---

## 12. Raw Data Integrity

The raw data stored in `data/raw/` is treated as immutable source truth.

* **File Counts Verified**:
  * Aadhaar: 5,292 files
  * PAN: 3,452 files
* **Audit Verification**:
  ```text
  RAW DATA MODIFIED: NO
  ```
  Pre- and post-audit SHA-256 hash checks confirmed zero modification or deletion of source data files.

---

## 13. What Was Completed on Day 1

* [x] Project architecture and directory structure established.
* [x] Virtual environment (`.venv`) configured with core analysis/dev packages.
* [x] Raw Aadhaar and PAN datasets placed and verified.
* [x] Read-only dataset audit tooling designed, implemented, and executed.
* [x] Audit reports generated detailing file counts, dimensions, geometries, and anomalies.
* [x] Leakage and duplicate issues identified and quantified in both datasets.
* [x] Local annotation visualization tool implemented and executed.
* [x] Semantic field meanings manually verified and mapped for Aadhaar (0–3) and PAN (0–3).
* [x] Configuration recorded in `configs/class_mapping.yaml`.
* [x] 69 tests written and passing.
* [x] Source data immutability verified.

---

## 14. What Remains Unfinished

The following areas are pending future phases:

* [ ] Leakage-free dataset splitting (grouping by original source image).
* [ ] Aadhaar source-level grouping and re-splitting into train/validation/test sets.
* [ ] PAN deduplication and creation of a clean validation and test split.
* [ ] Generation of clean processed datasets in `data/processed/`.
* [ ] Final decision and resolution on Aadhaar class 4.
* [ ] Finalization of unified annotation schema (`idocr.data.annotation_schema`).
* [ ] Source-to-unified annotation converters (`idocr.data.converters`).
* [ ] OCR text transcription dataset creation.
* [ ] Document classification model selection and training.
* [ ] Field text detection model selection and training.
* [ ] Text recognition (OCR) model selection and training.
* [ ] End-to-end evaluation pipeline benchmarking.
* [ ] Synthetic Driving Licence dataset generator implementation.
* [ ] Backend API service.
* [ ] Frontend UI.

---

## 15. Agent Responsibilities

### Claude Code (Previous Agent)
* Implemented the initial project scaffolding and interfaces.
* Built and executed the read-only dataset audit tooling.
* Audited the Aadhaar and PAN datasets and identified cross-split leakage.
* Implemented the local annotation visualization tool.
* Created and maintained the 69-test automated test suite.
* Created initial configuration files and documentation.

### Antigravity (Current Agent)
* Current coding agent responsible for continuing project execution under the user's direction.
* Consolidated and formalized Day 1 documentation from the completed project state.
* Responsible for implementing subsequent data processing, modeling, and pipeline stages.

### User (Project Owner)
* Project architecture and technical direction.
* Inspected visual contact sheets and manually determined annotation class meanings.
* Final authority on dataset splitting, schema definitions, and model decisions.

---

## 16. Current Project Position

```text
Day 1 completed:
Dataset discovery -> Audit -> Visualization -> Manual class identification -> Documentation

Current position:
Raw datasets understood and protected, but not yet cleaned/split for training.

Next major phase:
Create leakage-free processed datasets while keeping data/raw immutable.
```

---

## 17. Next Planned Phase

The immediate next step is the design and implementation of **leakage-free dataset splitting and processing**:
1. Group Aadhaar images by their underlying source document to ensure no augmented variants cross the train/validation/test boundaries.
2. Deduplicate PAN images and partition into clean train, validation, and test splits.
3. Write clean, converted datasets into `data/processed/` while preserving `data/raw/` untouched.
