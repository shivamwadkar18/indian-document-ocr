"""Read-only image integrity / metadata inspection."""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

#: EXIF tag id for Orientation.
_EXIF_ORIENTATION = 0x0112


@dataclass
class ImageInfo:
    path: Path
    ok: bool
    format: str | None = None
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    channels: int | None = None
    # EXIF orientation (1 = normal). Values 2-8 mean viewers rotate/flip the
    # stored pixels, which matters for how normalised label coordinates apply.
    exif_orientation: int | None = None
    sha256: str | None = None
    dhash: int | None = None  # 64-bit difference hash for near-duplicate detection
    error: str | None = None


def difference_hash(img: Image.Image, size: int = 8) -> int:
    """64-bit dHash: compares horizontally adjacent pixels of a 9x8 greyscale thumbnail."""
    small = img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    px = small.tobytes()  # mode "L": one byte per pixel, row-major
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def inspect_image(path: str | Path) -> ImageInfo:
    """Read an image's bytes once, verify it decodes, and record metadata + hashes.

    ``verify()`` catches truncated/corrupt files cheaply; the image is then
    reopened and fully loaded because ``verify()`` leaves the object unusable
    and misses some decode errors.
    """
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ImageInfo(path, False, error=f"{type(exc).__name__}: {exc}")
    digest = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            orientation = img.getexif().get(_EXIF_ORIENTATION)
            return ImageInfo(
                path,
                True,
                format=img.format,
                width=img.width,
                height=img.height,
                mode=img.mode,
                channels=len(img.getbands()),
                exif_orientation=int(orientation) if orientation is not None else None,
                sha256=digest,
                dhash=difference_hash(img),
            )
    except Exception as exc:  # noqa: BLE001 — any decode failure means "corrupted"
        return ImageInfo(path, False, sha256=digest, error=f"{type(exc).__name__}: {exc}")


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    s = sorted(values)
    return {"min": s[0], "max": s[-1], "median": s[len(s) // 2], "mean": round(sum(s) / len(s), 4)}


def _aspect_bucket(ratio: float) -> str:
    edges = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    lo = 0.0
    for hi in edges:
        if ratio < hi:
            return f"[{lo:.2f},{hi:.2f})"
        lo = hi
    return f">={edges[-1]:.2f}"


def summarize_images(
    infos: Iterable[ImageInfo], root: Path, *, min_side: int = 100, max_aspect: float = 3.0
) -> dict:
    """Aggregate image metadata. ``min_side`` / ``max_aspect`` define "unusual" sizes."""
    infos = list(infos)
    good = [i for i in infos if i.ok]
    sizes = Counter(f"{i.width}x{i.height}" for i in good)
    unusual = [
        {"path": i.path.relative_to(root).as_posix(), "size": f"{i.width}x{i.height}"}
        for i in good
        if min(i.width, i.height) < min_side or max(i.width / i.height, i.height / i.width) > max_aspect
    ]
    return {
        "count": len(infos),
        "readable": len(good),
        "corrupted": [
            {"path": i.path.relative_to(root).as_posix(), "error": i.error} for i in infos if not i.ok
        ],
        "formats": dict(Counter(i.format for i in good)),
        "modes": dict(Counter(i.mode for i in good)),
        "channels": dict(Counter(str(i.channels) for i in good)),
        "exif_orientation": dict(Counter(str(i.exif_orientation) for i in good)),
        "width": _stats([i.width for i in good]),
        "height": _stats([i.height for i in good]),
        "distinct_sizes": len(sizes),
        "most_common_sizes": dict(sizes.most_common(10)),
        "aspect_ratio_w_over_h": _stats([round(i.width / i.height, 4) for i in good]),
        "aspect_ratio_buckets": dict(sorted(Counter(_aspect_bucket(i.width / i.height) for i in good).items())),
        "unusual_sizes": {"min_side": min_side, "max_aspect": max_aspect, "count": len(unusual), "items": unusual},
    }
