"""Data tooling: discovery, audit, splitting, conversion, schema, preprocessing, augmentation, synthetic data."""

from idocr.data.annotation_schema import (
    DETECTOR_CLASSES,
    DOCUMENT_FIELDS,
    FIELD_NAME_TO_DETECTOR_CLASS,
    SCHEMA_VERSION,
    AnnotatedDocument,
    AnnotatedField,
    DocumentProvenance,
    FieldGeometry,
    FieldTranscription,
    FieldValidationRule,
    TranscriptionStatus,
    get_detector_class_id,
    get_detector_field_name,
    get_valid_fields,
    is_valid_field,
)
from idocr.data.detection import (
    DetectorDatasetBuilder,
    DetectorDatasetStats,
    DetectorDatasetValidator,
    DetectorValidationResult,
)
from idocr.data.discovery import discover_datasets, iter_files

__all__ = [
    "SCHEMA_VERSION",
    "DOCUMENT_FIELDS",
    "DETECTOR_CLASSES",
    "FIELD_NAME_TO_DETECTOR_CLASS",
    "TranscriptionStatus",
    "FieldGeometry",
    "FieldTranscription",
    "FieldValidationRule",
    "AnnotatedField",
    "DocumentProvenance",
    "AnnotatedDocument",
    "get_detector_class_id",
    "get_detector_field_name",
    "get_valid_fields",
    "is_valid_field",
    "discover_datasets",
    "iter_files",
    "DetectorDatasetBuilder",
    "DetectorDatasetStats",
    "DetectorDatasetValidator",
    "DetectorValidationResult",
]
