"""Local-only annotation visualization for manual class identification.

Reads raw datasets read-only (the verified Roboflow YOLO layout), renders
annotations onto downscaled copies in memory, and writes contact sheets plus
a metadata index under ``data/reports/visualizations/`` (git-ignored).

Class IDs are shown as ``class=<id>`` only; meanings are never inferred.
Nothing here calls external services or performs OCR.
"""

from idocr.data.visualization.dataset import LabeledImage, load_split
from idocr.data.visualization.geometry import polygon_to_pixels, scale_for_display, yolo_box_to_pixels
from idocr.data.visualization.sampling import Anomaly, find_anomalies, representative_sample

__all__ = [
    "Anomaly",
    "LabeledImage",
    "find_anomalies",
    "load_split",
    "polygon_to_pixels",
    "representative_sample",
    "scale_for_display",
    "yolo_box_to_pixels",
]
