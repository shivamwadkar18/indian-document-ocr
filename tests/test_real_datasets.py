"""Checks against the real Aadhaar / PAN datasets (skipped when not present).

These pin the behaviour verified by the audit so that a changed or re-exported
dataset is noticed. They read files only; immutability is asserted by hashing.
No image decoding here — the full audit (scripts/audit_dataset.py) covers that.
"""

from pathlib import Path

import pytest

from idocr.data.audit import YoloTxtInspector, detect_layout, hash_tree, match_by_stem, parse_yolo_file
from idocr.data.discovery import discover_datasets
from idocr.utils import ProjectPaths, load_default_config

PATHS = ProjectPaths.from_config(load_default_config())
SPLIT_CANDIDATES = load_default_config()["audit"]["split_name_candidates"]

# Verified by the audit on 2026-10-05.
EXPECTED = {
    "aadhaar": {"splits": {"train": 1852, "valid": 529, "test": 265}, "class_ids": {0, 1, 2, 3, 4}},
    "pan": {"splits": {"train": 1458, "valid": 268}, "class_ids": {0, 1, 2, 3}},
}


def _present(doc: str) -> bool:
    loc = discover_datasets(PATHS.raw, [doc])[0]
    return loc.exists and not loc.is_empty


@pytest.fixture(params=sorted(EXPECTED))
def doc(request):
    if not _present(request.param):
        pytest.skip(f"{request.param} dataset not present")
    return request.param


def test_discovery_finds_dataset(doc):
    assert discover_datasets(PATHS.raw, [doc])[0].file_count > 0


def test_layout_and_pairing(doc):
    root = PATHS.raw_dir(doc)
    layout = detect_layout(root, SPLIT_CANDIDATES)
    assert {s.name: None for s in layout.splits}.keys() == EXPECTED[doc]["splits"].keys()
    assert layout.unexpected == []
    for s in layout.splits:
        imgs = sorted(s.images_dir.glob("*.jpg"))
        lbls = sorted(s.labels_dir.glob("*.txt"))
        assert len(imgs) == len(lbls) == EXPECTED[doc]["splits"][s.name]
        m = match_by_stem(imgs, lbls)
        assert len(m.pairs) == len(imgs)
        assert not m.images_without_labels and not m.labels_without_images and not m.ambiguous_stems


def test_labels_are_yolo_and_parse(doc):
    root = PATHS.raw_dir(doc)
    before = hash_tree(root)
    labels = sorted(root.glob("*/labels/*.txt"))
    assert YoloTxtInspector().can_inspect(root, labels)
    class_ids = set()
    malformed = []
    for p in labels:
        f = parse_yolo_file(p, p.relative_to(root).as_posix())
        class_ids |= {a.class_id for a in f.annotations}
        malformed += [i for i in f.issues if i.problem != "empty_label_file"]
    assert class_ids == EXPECTED[doc]["class_ids"]
    assert malformed == []
    assert hash_tree(root) == before  # source untouched
