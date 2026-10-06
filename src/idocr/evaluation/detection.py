"""Text detection metrics.

Implemented: IoU, greedy-matched precision/recall at a fixed IoU threshold.
TODO: mAP (needs a decision on IoU thresholds and whether evaluation is
class-agnostic or per field class — depends on the audited labels).
TODO: polygon IoU if polygon labels are adopted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from idocr.types import BBox, TextRegion


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class DetectionScores:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_detections(
    predictions: Sequence[TextRegion], ground_truth: Sequence[BBox], iou_threshold: float = 0.5
) -> DetectionScores:
    """Greedy one-to-one matching, highest-confidence predictions first."""
    unmatched = list(range(len(ground_truth)))
    tp = 0
    for pred in sorted(predictions, key=lambda p: p.confidence, reverse=True):
        best, best_iou = None, iou_threshold
        for gi in unmatched:
            v = iou(pred.bbox, ground_truth[gi])
            if v >= best_iou:
                best, best_iou = gi, v
        if best is not None:
            unmatched.remove(best)
            tp += 1
    return DetectionScores(tp, len(predictions) - tp, len(unmatched))


def mean_average_precision(*args, **kwargs) -> float:
    raise NotImplementedError("mAP is defined after the detection label format is known.")
