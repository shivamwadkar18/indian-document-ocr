"""Coordinate conversion from normalised YOLO annotations to image pixels.

YOLO values are fractions of the stored image's width/height, so pixel
coordinates are computed against the image exactly as stored on disk (no EXIF
rotation is applied; the audit found no EXIF orientation tags).
"""

from __future__ import annotations

from typing import Sequence


def yolo_box_to_pixels(
    cx: float, cy: float, w: float, h: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Normalised centre/size box -> absolute ``(x1, y1, x2, y2)`` pixels."""
    return (
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    )


def polygon_to_pixels(
    points: Sequence[tuple[float, float]], width: int, height: int
) -> list[tuple[float, float]]:
    """Normalised polygon points -> absolute pixel points."""
    return [(x * width, y * height) for x, y in points]


def scale_for_display(width: int, height: int, max_side: int) -> float:
    """Uniform downscale factor so the longer side fits ``max_side`` (never upscales).

    A single factor for both axes preserves the original aspect ratio.
    """
    longest = max(width, height)
    return 1.0 if longest <= max_side else max_side / longest
