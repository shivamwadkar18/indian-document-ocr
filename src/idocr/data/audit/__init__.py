"""Read-only dataset audit infrastructure.

The audit NEVER modifies a dataset: files are only opened for reading, reports
are refused if they would be written inside the dataset directory, and every
source file is hashed before and after the run (``AuditError`` if anything changed).

Generic checks: structure, split layout, image integrity/dimensions/formats,
image/label pairing, exact and near-duplicate images, cross-split leakage via
source filenames, redacted label previews.

Format-specific checks come from registered ``LabelInspector``s. Currently:
``YoloTxtInspector`` (verified on the Aadhaar and PAN datasets).
"""

from idocr.data.audit.images import ImageInfo, inspect_image
from idocr.data.audit.integrity import compare_trees, hash_tree
from idocr.data.audit.labels import (
    LabelInspector,
    YoloTxtInspector,
    parse_yolo_file,
    redact_token,
    register_inspector,
)
from idocr.data.audit.layout import SplitLayout, detect_layout
from idocr.data.audit.matching import MatchResult, match_by_stem
from idocr.data.audit.report import AuditReport
from idocr.data.audit.runner import AuditError, run_audit
from idocr.data.audit.structure import TreeSummary, scan_tree
from idocr.data.audit.summary import render_summary, write_summary

__all__ = [
    "AuditError",
    "AuditReport",
    "ImageInfo",
    "LabelInspector",
    "MatchResult",
    "SplitLayout",
    "TreeSummary",
    "YoloTxtInspector",
    "compare_trees",
    "detect_layout",
    "hash_tree",
    "inspect_image",
    "match_by_stem",
    "parse_yolo_file",
    "redact_token",
    "register_inspector",
    "render_summary",
    "run_audit",
    "scan_tree",
    "write_summary",
]
