# Open decisions

Decisions deliberately left open until the datasets have been audited, or until
the project owner decides. Code locations carry matching `TODO` comments.

## Audited and Resolved

| Decision | Status | Where |
|----------|--------|-------|
| Label format(s) and image-label pairing | **Resolved** | `data/audit/labels.py`, `data/audit/matching.py` |
| Unified annotation schema (geometry, granularity, transcriptions, classes) | **Resolved** | [`src/idocr/data/annotation_schema.py`](file:///c:/Projects/indian-document-ocr/src/idocr/data/annotation_schema.py) |
| Field set per document type | **Resolved** | `src/idocr/data/annotation_schema.py`, `configs/class_mapping.yaml` |
| Converters per source format (YOLO -> unified schema) | **Resolved** | [`src/idocr/data/converters/yolo.py`](file:///c:/Projects/indian-document-ocr/src/idocr/data/converters/yolo.py) |

## Remaining Architecture Decisions

| Decision | Where |
|----------|-------|
| Preprocessing steps (scan vs photo, resolution, orientation) | `data/preprocessing/base.py` |
| Text normalisation for CER/WER (case, Unicode, scripts) | `evaluation/recognition.py` |
| mAP definition (thresholds, class-agnostic vs per-field) | `evaluation/detection.py` |

## Owner decisions

| Decision | Where |
|----------|-------|
| Detection architecture | `models/detector/`, `training/detection/` |
| Recognition architecture + character set (English / Hindi / regional) | `models/recognizer/`, `training/recognition/` |
| Classifier approach; source of `unknown` negatives | `models/classifier/`, `training/classification/` |
| Validation rules (PAN pattern, Aadhaar Verhoeff checksum, date checks) | `validation/` |
| Aadhaar number masking in outputs | `extraction/aadhaar.py` |
| Synthetic DL: fictional-ID scheme, name/address source, fonts, layouts | `data/synthetic/driving_license/` |
| Experiment tracking tool (if any) | — |
