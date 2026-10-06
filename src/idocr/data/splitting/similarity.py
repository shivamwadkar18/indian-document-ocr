"""Similarity and hashing utilities for image grouping and duplicate detection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from PIL import Image


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hexadecimal digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference_hash(img: Image.Image, size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) on a thumbnail."""
    small = img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    px = small.tobytes()
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def average_hash(img: Image.Image, size: int = 8) -> int:
    """Compute 64-bit average hash (aHash) on a thumbnail."""
    small = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    px = small.tobytes()
    avg = sum(px) / len(px)
    bits = 0
    for val in px:
        bits = (bits << 1) | (val > avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two integer bitmasks."""
    return (a ^ b).bit_count()


def find_near_duplicate_pairs(
    hashes: Mapping[str, int],
    threshold: int = 2,
) -> list[tuple[str, str, int]]:
    """Find pairs of image identifiers with Hamming distance <= threshold."""
    items = list(hashes.items())
    near_pairs: list[tuple[str, str, int]] = []
    for i in range(len(items)):
        k1, h1 = items[i]
        for j in range(i + 1, len(items)):
            k2, h2 = items[j]
            dist = (h1 ^ h2).bit_count()
            if dist <= threshold:
                near_pairs.append((k1, k2, dist))
    return sorted(near_pairs, key=lambda x: x[2])
