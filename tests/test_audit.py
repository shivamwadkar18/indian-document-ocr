"""Audit infrastructure tests on small temporary fixtures.

Fixture label files mirror the format verified in the real Aadhaar/PAN data
(Roboflow YOLO export: '<split>/images' + '<split>/labels', one .txt per image,
normalised 'class cx cy w h' lines, no trailing newline). They exist only to
exercise the parser and checks.
"""

import hashlib
import json

import pytest

from idocr.data.audit import (
    AuditError,
    YoloTxtInspector,
    compare_trees,
    detect_layout,
    hash_tree,
    inspect_image,
    match_by_stem,
    parse_yolo_file,
    redact_token,
    run_audit,
    write_summary,
)
from idocr.data.audit.naming import case_insensitive_collisions, roboflow_source_stem

RF = ".rf." + "0" * 32
RF2 = ".rf." + "1" * 32


def _snapshot(root):
    return {
        p.relative_to(root).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def dataset(tmp_path, make_image):
    root = tmp_path / "dataset"
    for split in ("train", "valid"):
        (root / split / "labels").mkdir(parents=True)
    # train: one good pair, one image w/o label, one orphan label, one corrupted image
    make_image(root / "train" / "images" / f"a_jpg{RF}.jpg", size=(100, 50), fmt="JPEG")
    (root / "train" / "labels" / f"a_jpg{RF}.txt").write_text("0 0.5 0.5 0.2 0.1\n1 0.3 0.3 0.1 0.1")
    make_image(root / "train" / "images" / f"b_jpg{RF}.jpg", fmt="JPEG")
    (root / "train" / "labels" / f"orphan_jpg{RF}.txt").write_text("0 0.5 0.5 0.2 0.1")
    (root / "train" / "images" / f"broken_jpg{RF}.jpg").write_bytes(b"not an image")
    (root / "train" / "labels" / f"broken_jpg{RF}.txt").write_text("0 0.5 0.5 0.2 0.1")
    # valid: same source name as train/a (leakage) — identical image bytes, conflicting label
    make_image(root / "valid" / "images" / f"a_jpg{RF2}.jpg", size=(100, 50), fmt="JPEG")
    (root / "valid" / "labels" / f"a_jpg{RF2}.txt").write_text("0 0.5 0.5 0.2 0.1")
    # unexpected top-level file
    (root / "README.extra").write_text("x")
    return root


# --- layout / naming / matching ---------------------------------------------


def test_detect_layout(dataset):
    layout = detect_layout(dataset, ["train", "valid", "test"])
    assert [s.name for s in layout.splits] == ["train", "valid"]
    assert all(s.name_is_known_split for s in layout.splits)
    assert layout.unexpected == ["README.extra"]


def test_roboflow_source_stem():
    assert roboflow_source_stem("0521_adhar_jpg.rf.0188ba6593c38a70e56de2509a244a4d") == "0521_adhar_jpg"
    assert roboflow_source_stem("plain_name") is None
    assert roboflow_source_stem("x.rf.NOTHEX") is None


def test_case_insensitive_collisions(tmp_path):
    from pathlib import Path

    paths = [Path("d/A.jpg"), Path("d/a.jpg"), Path("d/b.jpg"), Path("e/A.jpg")]
    assert case_insensitive_collisions(paths) == [["A.jpg", "a.jpg"]]


def test_match_by_stem(dataset):
    imgs = sorted((dataset / "train" / "images").iterdir())
    lbls = sorted((dataset / "train" / "labels").iterdir())
    m = match_by_stem(imgs, lbls)
    assert sorted(i.stem for i, _ in m.pairs) == [f"a_jpg{RF}", f"broken_jpg{RF}"]
    assert [p.stem for p in m.images_without_labels] == [f"b_jpg{RF}"]
    assert [p.stem for p in m.labels_without_images] == [f"orphan_jpg{RF}"]


# --- images -----------------------------------------------------------------


def test_inspect_image(dataset):
    good = inspect_image(dataset / "train" / "images" / f"a_jpg{RF}.jpg")
    assert good.ok and (good.width, good.height) == (100, 50)
    assert good.format == "JPEG" and good.channels == 3 and good.exif_orientation is None
    assert good.sha256 and good.dhash is not None
    bad = inspect_image(dataset / "train" / "images" / f"broken_jpg{RF}.jpg")
    assert not bad.ok and bad.error


# --- YOLO label parsing -----------------------------------------------------


def _parse(tmp_path, text, name="x.txt"):
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    return parse_yolo_file(p, name)


def test_parse_bbox_and_polygon_without_trailing_newline(tmp_path):
    f = _parse(tmp_path, "3 0.21 0.63 0.11 0.04\n4 0.1 0.5 0.2 0.5 0.3 0.6 0.1 0.5")
    assert not f.issues and not f.trailing_newline
    box, poly = f.annotations
    assert (box.class_id, box.kind, box.cx, box.w) == (3, "bbox", 0.21, 0.11)
    assert poly.kind == "polygon" and len(poly.points) == 4 and poly.points[0] == poly.points[-1]
    assert poly.w == pytest.approx(0.2) and poly.h == pytest.approx(0.1)


@pytest.mark.parametrize(
    "text, problem",
    [
        ("", "empty_label_file"),
        ("0 0.5 0.5 abc 0.1", "malformed_line"),
        ("0 0.5 0.5 0.1", "invalid_token_count"),
        ("0 0.1 0.2 0.3 0.4 0.5 0.6 0.7", "invalid_token_count"),  # odd number of polygon coords
        ("1.5 0.5 0.5 0.1 0.1", "invalid_class_id"),
        ("-1 0.5 0.5 0.1 0.1", "invalid_class_id"),
        ("0 1.5 0.5 0.1 0.1", "coord_out_of_range"),
        ("0 0.95 0.5 0.2 0.1", "bbox_outside_image"),
        ("0 0.5 0.5 0 0.1", "zero_or_negative_size"),
        ("0 0.1 0.1 0.2 0.2 0.3 0.3", "zero_area_polygon"),  # collinear points
        ("0 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1", "duplicate_annotation"),
        (b"\xff\xfe\x00", "encoding_error"),
    ],
)
def test_parse_detects_malformed(tmp_path, text, problem):
    f = _parse(tmp_path, text)
    assert problem in {i.problem for i in f.issues}


def test_yolo_inspector_can_inspect(tmp_path):
    good = tmp_path / "g.txt"
    good.write_text("0 0.5 0.5 0.1 0.1")
    text = tmp_path / "t.txt"
    text.write_text("name: something")
    insp = YoloTxtInspector()
    assert insp.can_inspect(tmp_path, [good])
    assert not insp.can_inspect(tmp_path, [good, text])
    assert not insp.can_inspect(tmp_path, [tmp_path / "x.json"])


def test_yolo_inspector_stats(tmp_path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.2 0.1\n1 0.3 0.3 0.1 0.1")
    (tmp_path / "b.txt").write_text("0 0.5 0.5 0.002 0.1")
    stats = YoloTxtInspector(tiny_box_px=4).inspect(
        tmp_path, sorted(tmp_path.glob("*.txt")), {"a": (100, 50), "b": (1000, 500)}
    )
    assert stats["class_ids"] == [0, 1]
    assert stats["class_counts"] == {"0": 2, "1": 1}
    assert stats["per_class"]["1"]["instances_per_file"] == {"0": 1, "1": 1}
    assert stats["per_class"]["0"]["meaning"] is None
    assert stats["transcriptions_present"] is False
    assert stats["tiny_boxes"]["count"] == 1  # 0.002 * 1000 = 2 px
    assert stats["file_format"]["no_trailing_newline"] == 2


# --- redaction --------------------------------------------------------------


@pytest.mark.parametrize(
    "token, expected",
    [("3", "3"), ("0.21015625", "0.21015625"), ("1234", "####"), ("ABCDE1234F", "XXXXX####X"), ("Ram", "Xxx")],
)
def test_redact_token(token, expected):
    assert redact_token(token) == expected


# --- full run ---------------------------------------------------------------


def test_run_audit_report_and_immutability(dataset, tmp_path):
    before = _snapshot(dataset)
    reports = tmp_path / "reports"
    report = run_audit(dataset, reports, name="demo")
    assert _snapshot(dataset) == before
    assert report.sections["source_integrity"]["source_data_modified"] is False

    train = report.sections["splits"]["train"]
    assert (train["image_files"], train["label_files"]) == (3, 3)
    assert train["pairing"]["pairs"] == 2
    assert len(train["images"]["corrupted"]) == 1
    assert train["annotations"]["class_counts"] == {"0": 3, "1": 1}
    assert report.sections["label_format"]["inspector"] == "yolo_txt"
    assert report.sections["pairing_rule"]["status"].startswith("holds with exceptions")

    sn = report.sections["source_naming"]
    assert sn["source_names_in_multiple_splits"] == 1  # a_jpg in train + valid
    dup = report.sections["duplicates"]
    assert dup["exact_duplicate_groups_across_splits"] == 1
    assert dup["label_consistency"]["groups_with_conflicting_labels"] == 1

    problems = {i["problem"] for i in report.sections["issues"]}
    assert {"corrupted_image", "image_without_label", "label_without_image", "unexpected_entry",
            "exact_duplicate_across_splits", "duplicate_images_conflicting_labels"} <= problems

    # previews are redacted and contain only numeric tokens here
    previews = report.sections["label_previews"][".txt"]
    assert all(p["redacted"] for p in previews.values())

    data = json.loads((reports / "demo_report.json").read_text(encoding="utf-8"))
    assert data["dataset_name"] == "demo"
    summary = write_summary(reports).read_text(encoding="utf-8")
    assert "SOURCE DATA MODIFIED: NO" in summary
    assert "Meaning not determinable from annotation files alone." in summary


def test_run_audit_clean_dataset_pairing_verified(tmp_path, make_image):
    root = tmp_path / "ds"
    (root / "train" / "labels").mkdir(parents=True)
    make_image(root / "train" / "images" / "x.jpg", fmt="JPEG")
    (root / "train" / "labels" / "x.txt").write_text("0 0.5 0.5 0.5 0.5")
    report = run_audit(root)
    assert report.sections["pairing_rule"]["status"] == "verified: holds for every file"
    assert report.sections["issues"] == []


def test_integrity_detects_changes(tmp_path):
    (tmp_path / "a.txt").write_text("1")
    before = hash_tree(tmp_path)
    (tmp_path / "a.txt").write_text("2")
    (tmp_path / "b.txt").write_text("new")
    result = compare_trees(before, hash_tree(tmp_path))
    assert result.modified and result.changed == ["a.txt"] and result.added == ["b.txt"]


def test_run_audit_refuses_reports_inside_dataset(dataset):
    with pytest.raises(AuditError):
        run_audit(dataset, dataset / "reports")
    assert not (dataset / "reports").exists()


def test_run_audit_missing_dataset(tmp_path):
    with pytest.raises(AuditError):
        run_audit(tmp_path / "nope")
