import pytest

from idocr.evaluation.detection import iou, match_detections
from idocr.evaluation.end_to_end import document_accuracy, field_accuracy
from idocr.evaluation.recognition import (
    character_error_rate,
    exact_match_accuracy,
    levenshtein,
    word_error_rate,
)
from idocr.types import BBox, TextRegion


def test_iou():
    assert iou(BBox(0, 0, 10, 10), BBox(0, 0, 10, 10)) == 1.0
    assert iou(BBox(0, 0, 10, 10), BBox(20, 20, 30, 30)) == 0.0
    assert iou(BBox(0, 0, 10, 10), BBox(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_match_detections():
    gt = [BBox(0, 0, 10, 10), BBox(20, 20, 30, 30)]
    preds = [TextRegion(BBox(0, 0, 10, 10), 0.9), TextRegion(BBox(50, 50, 60, 60), 0.8)]
    s = match_detections(preds, gt)
    assert (s.true_positives, s.false_positives, s.false_negatives) == (1, 1, 1)
    assert s.precision == 0.5 and s.recall == 0.5


def test_text_metrics():
    assert levenshtein("kitten", "sitting") == 3
    assert character_error_rate(["abc"], ["abd"]) == pytest.approx(1 / 3)
    assert word_error_rate(["a b c"], ["a x c"]) == pytest.approx(1 / 3)
    assert exact_match_accuracy(["a", "b"], ["a", "c"]) == 0.5
    with pytest.raises(ValueError):
        character_error_rate(["a"], [])


def test_end_to_end_metrics():
    preds = [{"name": "A", "id": "1"}, {"name": "B", "id": "9"}]
    refs = [{"name": "A", "id": "1"}, {"name": "B", "id": "2"}]
    assert field_accuracy(preds, refs) == {"id": 0.5, "name": 1.0}
    assert document_accuracy(preds, refs) == 0.5
