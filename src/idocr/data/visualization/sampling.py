"""Deterministic, representative sampling and data-driven anomaly selection."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from idocr.data.visualization.dataset import LabeledImage

#: A class present in at least this fraction of files is a "core" class.
CORE_CLASS_MIN_FRACTION = 0.5
#: A class present in fewer than this fraction of files is "rare".
RARE_CLASS_MAX_FRACTION = 0.1


def representative_sample(items: Sequence[LabeledImage], n: int, key: str) -> list[LabeledImage]:
    """Pick up to ``n`` items spread across distinct source images.

    Items are grouped by source name (augmented copies share one), groups are
    visited in a seeded random order and one item is taken per group per
    round. So ``n`` samples come from ``n`` different sources whenever
    possible. ``key`` seeds the RNG (str seeds are hashed deterministically).
    """
    if n <= 0 or not items:
        return []
    rng = random.Random(key)
    groups: dict[str, list[LabeledImage]] = defaultdict(list)
    for item in sorted(items, key=lambda i: (i.split, i.filename)):
        groups[item.source].append(item)
    order = sorted(groups)
    rng.shuffle(order)
    for name in order:
        rng.shuffle(groups[name])
    picked: list[LabeledImage] = []
    depth = 0
    while len(picked) < n and any(len(groups[g]) > depth for g in order):
        for g in order:
            if len(groups[g]) > depth:
                picked.append(groups[g][depth])
                if len(picked) == n:
                    break
        depth += 1
    return picked


@dataclass
class Anomaly:
    kind: str
    description: str
    items: list[LabeledImage] = field(default_factory=list)


def class_presence(items: Sequence[LabeledImage]) -> dict[int, float]:
    """Fraction of files containing each class."""
    if not items:
        return {}
    present = Counter(c for i in items for c in i.class_counts)
    return {c: present[c] / len(items) for c in sorted(present)}


def find_anomalies(items: Sequence[LabeledImage]) -> list[Anomaly]:
    """Data-driven anomaly groups. Every kind is always returned (possibly empty)."""
    presence = class_presence(items)
    core = sorted(c for c, f in presence.items() if f >= CORE_CLASS_MIN_FRACTION)
    rare = sorted(c for c, f in presence.items() if f < RARE_CLASS_MAX_FRACTION)
    counts = Counter(len(i.annotations) for i in items if i.label_path is not None)
    modal = counts.most_common(1)[0][0] if counts else None

    def where(pred) -> list[LabeledImage]:
        return [i for i in items if pred(i)]

    out = [
        Anomaly("empty_label_file", "label file exists but has no annotations",
                where(lambda i: any(x.problem == "empty_label_file" for x in i.issues))),
        Anomaly("missing_label_file", "image without a label file",
                where(lambda i: i.label_path is None)),
        Anomaly("malformed_annotation", "label file with parse problems (malformed, out of range, ...)",
                where(lambda i: any(x.problem not in ("empty_label_file", "missing_label_file") for x in i.issues))),
        Anomaly("missing_core_class", f"missing at least one core class {core} (present in >=50% of files)",
                where(lambda i: i.label_path is not None and any(c not in i.class_counts for c in core))),
        Anomaly("duplicate_class_instances", "a class ID appears more than once in the image",
                where(lambda i: any(n > 1 for n in i.class_counts.values()))),
        Anomaly("polygon_annotation", "contains a polygon annotation line",
                where(lambda i: any(a.kind == "polygon" for a in i.annotations))),
    ]
    duplicated = sorted({c for i in items for c, n in i.class_counts.items() if n > 1})
    for c in duplicated:
        out.append(Anomaly(f"duplicate_class_{c}", f"class={c} appears more than once in the image",
                           where(lambda i, c=c: i.class_counts.get(c, 0) > 1)))
    for c in rare:
        out.append(Anomaly(f"rare_class_{c}", f"contains class={c} (present in <10% of files)",
                           where(lambda i, c=c: c in i.class_counts)))
    for k in sorted(counts):
        if k != modal:
            out.append(Anomaly(f"annotation_count_{k}", f"exactly {k} annotations (typical: {modal})",
                               where(lambda i, k=k: i.label_path is not None and len(i.annotations) == k)))
    return out
