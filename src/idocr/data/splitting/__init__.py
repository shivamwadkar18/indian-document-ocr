"""Dataset splitting and grouping package for leakage-free train/validation/test sets."""

from idocr.data.splitting.grouping import (
    ConfidenceLevel,
    GroupingMethod,
    ImageRecord,
    SourceGroup,
    group_dataset,
    scan_image_records,
)
from idocr.data.splitting.manifest import (
    create_manifest_records,
    load_manifest_csv,
    write_manifest_csv,
)
from idocr.data.splitting.processor import build_processed_dataset
from idocr.data.splitting.similarity import (
    average_hash,
    difference_hash,
    find_near_duplicate_pairs,
    hamming_distance,
    sha256_file,
)
from idocr.data.splitting.splitter import split_source_groups
from idocr.data.splitting.validation import (
    ValidationReport,
    validate_processed_splits,
)

__all__ = [
    "ConfidenceLevel",
    "GroupingMethod",
    "ImageRecord",
    "SourceGroup",
    "group_dataset",
    "scan_image_records",
    "create_manifest_records",
    "load_manifest_csv",
    "write_manifest_csv",
    "build_processed_dataset",
    "difference_hash",
    "average_hash",
    "hamming_distance",
    "sha256_file",
    "find_near_duplicate_pairs",
    "split_source_groups",
    "ValidationReport",
    "validate_processed_splits",
]
