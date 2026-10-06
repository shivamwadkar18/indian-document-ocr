"""Image <-> label pairing heuristic.

HEURISTIC ONLY: pairs files whose filename stems are equal (``a/x.jpg`` with
``b/x.txt``). Many datasets follow this convention, but some do not (e.g. one
JSON file for all images, CSV manifests). The audit report labels this result
as a heuristic; the real pairing rule is decided after manual inspection.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class MatchResult:
    pairs: list[tuple[Path, Path]] = field(default_factory=list)
    images_without_labels: list[Path] = field(default_factory=list)
    labels_without_images: list[Path] = field(default_factory=list)
    # stems shared by more than one image or more than one label file
    ambiguous_stems: list[str] = field(default_factory=list)


def match_by_stem(images: Iterable[Path], labels: Iterable[Path]) -> MatchResult:
    by_stem_img: dict[str, list[Path]] = defaultdict(list)
    by_stem_lbl: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        by_stem_img[p.stem].append(p)
    for p in labels:
        by_stem_lbl[p.stem].append(p)

    result = MatchResult()
    for stem in sorted(set(by_stem_img) | set(by_stem_lbl)):
        imgs, lbls = by_stem_img.get(stem, []), by_stem_lbl.get(stem, [])
        if len(imgs) > 1 or len(lbls) > 1:
            result.ambiguous_stems.append(stem)
        if imgs and lbls:
            result.pairs.extend((i, l) for i in imgs for l in lbls)
        elif imgs:
            result.images_without_labels.extend(imgs)
        else:
            result.labels_without_images.extend(lbls)
    return result
