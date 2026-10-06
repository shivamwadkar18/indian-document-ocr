"""Detection dataset preparation and validation package."""

from idocr.data.detection.builder import DetectorDatasetBuilder, DetectorDatasetStats
from idocr.data.detection.validation import DetectorDatasetValidator, DetectorValidationResult

__all__ = [
    "DetectorDatasetBuilder",
    "DetectorDatasetStats",
    "DetectorDatasetValidator",
    "DetectorValidationResult",
]
