# Day 2 — Data Preparation & Detector Dataset

## Overview

Day 2 completed the dataset engineering foundation for the `indian-document-ocr` pipeline. All source datasets were split without data leakage, an unresolved Aadhaar annotation class was investigated and resolved, a canonical unified annotation schema and converter were established, and a single, common field-detector training dataset combining Aadhaar and PAN was produced and validated.

---

## Key Achievements & Decisions

### 1. Leakage-Free Source-Group Splitting
- **Problem**: Raw Aadhaar and PAN datasets contained augmentations, crops, and duplicates of identical underlying source documents across multiple images. A random image-level split caused severe data leakage across train/valid/test splits.
- **Solution**: Implemented source-group discovery (`src/idocr/data/splitting/`) grouping images by perceptual hashes, base identifiers, and duplicate clusters. Splits were executed deterministically at the source-group level.
- **Outcome**:
  - **Aadhaar**: 2,646 images across 2,476 source groups.
  - **PAN**: 1,726 images across 1,600 source groups.
  - **Zero Cross-Split Leakage**: 0 source groups or images cross split boundaries.
  - **Raw Data Immutability**: All original raw files in `data/raw/` remained 100% untouched.
  - Processed splits created under `data/processed/aadhaar/` and `data/processed/pan/` with split manifests in `data/processed/manifests/`.

### 2. Aadhaar Class 4 Investigation & Resolution
- **Investigation**: Analyzed 87 occurrences of Aadhaar YOLO Class 4 across 86 images. Class 4 was found to be highly inconsistent annotation noise (encompassing non-standard objects, stray marks, partial cards, and background regions).
- **Decision**:
  - Excluded from all ML detector training targets (producing 0 detector labels).
  - Preserved unmodified in raw and processed source annotation files.
  - No speculative semantic meaning or pseudo-class was assigned.

### 3. Unified Annotation Schema
- Established a single, document-independent schema (`src/idocr/data/annotation_schema.py`) to standardize representations across Aadhaar, PAN, and future Driving Licence documents.
- **Components**:
  - **Document Metadata**: Document type (`DocumentType.AADHAAR`, `PAN`, `DRIVING_LICENSE`), schema version (`1.0.0`), document dimensions.
  - **Field Representation**: Canonical field names, normalized & absolute bounding boxes / polygons (`FieldGeometry`).
  - **OCR Transcription**: Transcription string, confidence, and status enum (`TranscriptionStatus.MISSING` for pure detection datasets).
  - **Provenance**: Source dataset name, split partition, source group ID, and image SHA-256 hash.

### 4. YOLO → Unified Annotation Converter
- Built `YoloDatasetConverter` (`src/idocr/data/converters/yolo.py`) to convert legacy YOLO coordinate files into unified `AnnotatedDocument` structures.
- Verified across all 4,372 processed images (10,310 Aadhaar annotations + 6,879 PAN annotations) with 0 errors, properly excluding 87 Class 4 noise instances without fabricating OCR text.

### 5. Unified Detector Class Vocabulary
Established a shared 6-class vocabulary (`configs/detection_classes.yaml`) to train a single, multi-document field detector:

| Class ID | Canonical Field Name | Applicable Documents | Description |
| :---: | :--- | :--- | :--- |
| `0` | `name` | Aadhaar, PAN, future DL | Full name of document holder (shared) |
| `1` | `date_of_birth` | Aadhaar, PAN, future DL | Date or year of birth (shared) |
| `2` | `gender` | Aadhaar | Gender indicator |
| `3` | `aadhaar_number` | Aadhaar | 12-digit Aadhaar number |
| `4` | `pan_number` | PAN | 10-character alphanumeric PAN ID |
| `5` | `fathers_name` | PAN, future DL | Father's / Parent's name |

*Note: Future Driving Licence fields (`license_number`, `issue_date`, `expiry_date`, `address`, `vehicle_classes`) will append cleanly starting at Class ID 6.*

### 6. Common Detector-Ready Dataset
Generated the unified YOLO-format dataset combining Aadhaar and PAN under `data/processed/detector/` via `scripts/generate_detector_dataset.py`.

---

## Final Detector Dataset Metrics

### Split Distribution

| Split | Aadhaar Images | PAN Images | Total Images | Total Labels | Detector Targets | Empty Labels |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`train`** | 1,852 | 1,208 | **3,060** | **3,060** | **11,935** | 5 |
| **`valid`** | 396 | 258 | **654** | **654** | **2,555** | 4 |
| **`test`** | 398 | 260 | **658** | **658** | **2,612** | 0 |
| **Total** | **2,646** | **1,726** | **4,372** | **4,372** | **17,102** | **9** |

### Target Counts by Class

| Class ID | Field Name | Target Count |
| :---: | :--- | ---: |
| `0` | `name` | 4,258 |
| `1` | `date_of_birth` | 4,265 |
| `2` | `gender` | 2,554 |
| `3` | `aadhaar_number` | 2,606 |
| `4` | `pan_number` | 1,722 |
| `5` | `fathers_name` | 1,697 |
| **Total** | | **17,102** |

### Class 4 & Empty Label Handling
- **Aadhaar Class 4 Targets Excluded**: Exactly **87** instances.
- **Empty-Label Images**: Exactly **9** images (1 raw Aadhaar image with 0 annotations + 8 Aadhaar images where Class 4 was the only annotation) are preserved with empty `.txt` files as valid background training samples.

---

## Validation Summary

Validation suite executed via `DetectorDatasetValidator` (`src/idocr/data/detection/validation.py`) and recorded in `data/reports/detector_dataset_report.md`:

- **1:1 Image/Label Pairing**: **PASSED** (4,372 / 4,372).
- **Class ID Restriction (0–5)**: **PASSED** (no unmapped or out-of-range classes).
- **Normalized Box Geometry**: **PASSED** (all coordinates valid `[0, 1]`).
- **Cross-Split Leakage**: **PASSED** (0 source group or image overlap across splits).
- **Source Integrity**: **PASSED** (`data/raw/`, `data/processed/aadhaar/`, `data/processed/pan/` 100% unchanged).
- **Automated Test Suite**: **107 / 107 tests passing**.

---

## Day 2 Outcome & Next Steps

Aadhaar and PAN datasets are now consolidated into a verified, leakage-free, detector-ready dataset (`data/processed/detector/` with `data.yaml`).

The data engineering preparation is complete. The project is ready for **Day 3: Detector Model Development & Training** by Claude Code.
