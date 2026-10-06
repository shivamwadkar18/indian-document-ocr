"""Visualization workflow tests on temporary fixtures (solid-colour images, YOLO-format labels)."""

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from idocr.data.visualization import (
    find_anomalies,
    load_split,
    polygon_to_pixels,
    representative_sample,
    scale_for_display,
    yolo_box_to_pixels,
)
from idocr.data.visualization.dataset import LabeledImage
from idocr.data.visualization.render import class_color, render_annotated
from idocr.data.visualization.workflow import (
    VisualizationError,
    check_output_dir,
    visualize_split,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def rf(name: str, n: int) -> str:
    """Roboflow-style stem: <source>.rf.<32 hex>."""
    return f"{name}.rf.{n:032x}"


@pytest.fixture
def dataset(tmp_path, make_image):
    root = tmp_path / "raw" / "demo"
    labels = root / "train" / "labels"
    labels.mkdir(parents=True)
    std = "0 0.5 0.8 0.4 0.1\n1 0.6 0.4 0.2 0.1\n2 0.4 0.5 0.1 0.05\n3 0.4 0.3 0.2 0.07"
    files = {
        # 3 sources with several augmented copies each, all "normal"
        **{rf(f"src{s}_jpg", s * 10 + k): std for s in range(3) for k in range(4)},
        rf("dup0_jpg", 100): std + "\n0 0.5 0.9 0.4 0.05",            # class 0 twice (5 annotations)
        rf("missing3_jpg", 101): "0 0.5 0.8 0.4 0.1\n1 0.6 0.4 0.2 0.1\n2 0.4 0.5 0.1 0.05",
        rf("poly_jpg", 102): std + "\n4 0.1 0.5 0.2 0.5 0.3 0.6 0.1 0.5",  # rare class 4 as polygon
        rf("empty_jpg", 103): "",
        rf("bad_jpg", 104): "0 0.5 0.5 abc 0.1",
    }
    for stem, text in files.items():
        make_image(root / "train" / "images" / f"{stem}.jpg", size=(200, 100), fmt="JPEG")
        (labels / f"{stem}.txt").write_text(text)
    make_image(root / "train" / "images" / f"{rf('nolabel_jpg', 105)}.jpg", size=(200, 100), fmt="JPEG")
    return root


# --- geometry ---------------------------------------------------------------


def test_yolo_box_to_pixels():
    assert yolo_box_to_pixels(0.5, 0.5, 0.2, 0.4, 1000, 500) == pytest.approx((400, 150, 600, 350))
    assert yolo_box_to_pixels(0.1, 0.1, 0.2, 0.2, 640, 640) == pytest.approx((0, 0, 128, 128))


def test_polygon_to_pixels():
    assert polygon_to_pixels([(0.1, 0.5), (1.0, 0.0)], 200, 100) == pytest.approx([(20, 50), (200, 0)])


def test_scale_for_display_preserves_aspect_and_never_upscales():
    assert scale_for_display(2000, 1000, 640) == pytest.approx(0.32)
    assert scale_for_display(300, 200, 640) == 1.0


# --- parsing / loading ------------------------------------------------------


def test_load_split_parses_classes_and_handles_missing_empty_malformed(dataset):
    items = {i.image_path.stem.split(".rf.")[0]: i for i in load_split(dataset, "train")}
    assert items["src0_jpg"].class_counts == {0: 1, 1: 1, 2: 1, 3: 1}
    assert items["dup0_jpg"].class_counts[0] == 2
    assert items["poly_jpg"].annotation_types == ["bbox", "polygon"]
    assert items["nolabel_jpg"].label_path is None
    assert {x.problem for x in items["nolabel_jpg"].issues} == {"missing_label_file"}
    assert {x.problem for x in items["empty_jpg"].issues} == {"empty_label_file"}
    assert "malformed_line" in {x.problem for x in items["bad_jpg"].issues}
    assert items["src0_jpg"].source == "src0_jpg"


def test_load_split_missing_split(dataset):
    with pytest.raises(FileNotFoundError):
        load_split(dataset, "test")


# --- sampling ---------------------------------------------------------------


def test_representative_sample_is_deterministic(dataset):
    items = load_split(dataset, "train")
    a = [i.filename for i in representative_sample(items, 5, "42:x")]
    b = [i.filename for i in representative_sample(list(reversed(items)), 5, "42:x")]
    c = [i.filename for i in representative_sample(items, 5, "7:x")]
    assert a == b
    assert a != c or len(items) <= 5


def test_representative_sample_spreads_across_sources(dataset):
    items = [i for i in load_split(dataset, "train") if i.source.startswith("src")]
    picked = representative_sample(items, 3, "k")
    assert len({i.source for i in picked}) == 3  # one per source before repeating
    assert len(representative_sample(items, 100, "k")) == len(items)
    assert representative_sample(items, 0, "k") == []


def test_find_anomalies(dataset):
    found = {a.kind: {i.source for i in a.items} for a in find_anomalies(load_split(dataset, "train"))}
    assert found["empty_label_file"] == {"empty_jpg"}
    assert found["missing_label_file"] == {"nolabel_jpg"}
    assert found["malformed_annotation"] == {"bad_jpg"}
    assert found["duplicate_class_instances"] == {"dup0_jpg"}
    assert found["duplicate_class_0"] == {"dup0_jpg"}
    assert "duplicate_class_1" not in found  # per-class kinds only for classes that repeat
    assert found["polygon_annotation"] == {"poly_jpg"}
    assert found["rare_class_4"] == {"poly_jpg"}
    assert {"missing3_jpg", "empty_jpg", "bad_jpg"} <= found["missing_core_class"]
    assert found["annotation_count_3"] == {"missing3_jpg"}
    assert found["annotation_count_5"] == {"dup0_jpg", "poly_jpg"}


def test_find_anomalies_reports_empty_kinds():
    item = LabeledImage("train", Path("x.jpg"), Path("x.txt"))
    kinds = {a.kind: a.items for a in find_anomalies([item])}
    assert kinds["polygon_annotation"] == [] and kinds["missing_label_file"] == []


# --- rendering --------------------------------------------------------------


def test_render_annotated_draws_and_downscales(dataset, make_image):
    item = next(i for i in load_split(dataset, "train") if i.source == "poly_jpg")
    big = dataset / "big.jpg"
    Image.new("RGB", (2000, 1000), (255, 255, 255)).save(big)
    item.image_path = big
    out = render_annotated(item, 640, highlight_class=0)
    assert out.size == (640, 320)  # aspect preserved
    # top-left corner of the class-0 box (0.3*640, 0.75*320) carries the class colour
    assert out.getpixel((round(0.3 * 640) + 1, round(0.75 * 320) + 6)) == class_color(0)


# --- end to end -------------------------------------------------------------


def _snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_visualize_split_and_report(dataset, tmp_path):
    before = _snapshot(dataset)
    out = tmp_path / "viz"
    index = visualize_split("demo", dataset, "train", out, samples_per_class=2, mixed_samples=4, seed=1)
    assert _snapshot(dataset) == before  # raw untouched

    split_dir = out / "demo" / "train"
    files = {p.name for p in split_dir.iterdir()}
    assert {"class_0.jpg", "class_4.jpg", "mixed.jpg", "anomaly_polygon_annotation.jpg", "index.json"} <= files
    assert index["anomalies"]["missing_label_file"]["found"] == 1
    class0 = next(s for s in index["sheets"] if s["file"] == "class_0.jpg")
    assert len(class0["tiles"]) == 2 and all("0" in t["class_counts"] for t in class0["tiles"])

    # rerun with same seed -> identical tile selection; stale sheets cleaned
    (split_dir / "anomaly_polygon_annotation.jpg").unlink()
    again = visualize_split("demo", dataset, "train", out, samples_per_class=2, mixed_samples=4, seed=1)
    assert [t["source_filename"] for s in again["sheets"] for t in s["tiles"]] == \
           [t["source_filename"] for s in index["sheets"] for t in s["tiles"]]

    report = write_report(out).read_text(encoding="utf-8")
    assert "| Class ID | Annotations | Files with class | Sample tiles | Visual samples | My interpretation |" in report
    for line in report.splitlines():
        if line.startswith("| 0 |") or line.startswith("| 4 |"):
            assert line.rstrip().endswith("|  |")  # interpretation column left blank
    assert json.loads((split_dir / "index.json").read_text())["dataset"] == "demo"


def test_missing_anomaly_reported_not_failed(tmp_path, make_image):
    root = tmp_path / "raw" / "clean"
    (root / "train" / "labels").mkdir(parents=True)
    make_image(root / "train" / "images" / "a.jpg", fmt="JPEG")
    (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.5 0.5")
    out = tmp_path / "viz"
    index = visualize_split("clean", root, "train", out)
    assert index["anomalies"]["polygon_annotation"] == {
        "description": "contains a polygon annotation line", "found": 0, "visualized": 0, "sheet": None}
    assert not (out / "clean" / "train" / "anomaly_polygon_annotation.jpg").exists()
    assert "| train | polygon_annotation |" in write_report(out).read_text(encoding="utf-8")
    assert "Not found in this split" in write_report(out).read_text(encoding="utf-8")


def test_check_output_dir_refuses_raw(tmp_path):
    raw = tmp_path / "raw"
    with pytest.raises(VisualizationError):
        check_output_dir(raw / "aadhaar" / "viz", [raw])
    check_output_dir(tmp_path / "reports" / "viz", [raw])


def test_class_mapping_template_is_placeholder_shaped():
    mapping = yaml.safe_load((REPO_ROOT / "configs" / "class_mapping.yaml").read_text(encoding="utf-8"))
    assert sorted(mapping["aadhaar"]) == [0, 1, 2, 3, 4]
    assert sorted(mapping["pan"]) == [0, 1, 2, 3]
    for classes in mapping.values():
        for entry in classes.values():
            assert set(entry) == {"name", "description"}
