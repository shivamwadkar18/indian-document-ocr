"""Tests for dataset splitting, source grouping, and leakage-free partitioning."""

from pathlib import Path

import pytest

from idocr.data.audit.integrity import hash_tree
from idocr.data.splitting.grouping import (
    ImageRecord,
    SourceGroup,
    group_dataset,
    scan_image_records,
)
from idocr.data.splitting.manifest import (
    create_manifest_records,
    load_manifest_csv,
    write_manifest_csv,
)
from idocr.data.splitting.similarity import (
    difference_hash,
    hamming_distance,
    sha256_file,
)
from idocr.data.splitting.splitter import split_source_groups
from idocr.data.splitting.validation import validate_processed_splits


def _dummy_image_record(
    stem: str,
    raw_split: str = "train",
    rf_source: str | None = None,
    sha: str = "dummy_sha",
    dhash: int = 0,
    width: int = 640,
    height: int = 640,
) -> ImageRecord:
    return ImageRecord(
        path=Path(f"{raw_split}/images/{stem}.jpg"),
        rel_path=f"{raw_split}/images/{stem}.jpg",
        raw_split=raw_split,
        stem=stem,
        roboflow_source=rf_source,
        sha256=sha,
        dhash=dhash,
        width=width,
        height=height,
        label_path=Path(f"{raw_split}/labels/{stem}.txt"),
        annotation_count=4,
    )


def test_hamming_distance():
    assert hamming_distance(0, 0) == 0
    assert hamming_distance(0b1111, 0b0000) == 4
    assert hamming_distance(0b1010, 0b1001) == 2


def test_split_source_groups_deterministic():
    groups = [
        SourceGroup(group_id=f"grp_{i:03d}", document_type="aadhaar", members=[_dummy_image_record(f"img_{i}")])
        for i in range(100)
    ]
    split_1 = split_source_groups(groups, ratios=(0.7, 0.15, 0.15), seed=42)
    split_2 = split_source_groups(groups, ratios=(0.7, 0.15, 0.15), seed=42)

    assert [g.group_id for g in split_1["train"]] == [g.group_id for g in split_2["train"]]
    assert [g.group_id for g in split_1["valid"]] == [g.group_id for g in split_2["valid"]]
    assert [g.group_id for g in split_1["test"]] == [g.group_id for g in split_2["test"]]


def test_split_source_groups_isolation():
    # Multi-member groups
    g1 = SourceGroup(
        group_id="grp_001",
        document_type="aadhaar",
        members=[
            _dummy_image_record("img_1_aug1", raw_split="train"),
            _dummy_image_record("img_1_aug2", raw_split="valid"),
            _dummy_image_record("img_1_aug3", raw_split="test"),
        ],
    )
    g2 = SourceGroup(
        group_id="grp_002",
        document_type="aadhaar",
        members=[
            _dummy_image_record("img_2_aug1", raw_split="train"),
            _dummy_image_record("img_2_aug2", raw_split="valid"),
        ],
    )
    g3 = SourceGroup(
        group_id="grp_003",
        document_type="aadhaar",
        members=[_dummy_image_record("img_3", raw_split="test")],
    )

    splits = split_source_groups([g1, g2, g3], ratios=(0.7, 0.15, 0.15), seed=42)

    # Verify no group appears in multiple splits
    train_ids = {g.group_id for g in splits["train"]}
    valid_ids = {g.group_id for g in splits["valid"]}
    test_ids = {g.group_id for g in splits["test"]}

    assert len(train_ids & valid_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(valid_ids & test_ids) == 0


def test_manifest_roundtrip(tmp_path: Path):
    g1 = SourceGroup(
        group_id="grp_001",
        document_type="pan",
        members=[
            _dummy_image_record("img_1", raw_split="train", sha="hash1", dhash=123),
        ],
        grouping_method="single_source",
        confidence="unmatched",
    )
    splits = {"train": [g1], "valid": [], "test": []}
    csv_path = tmp_path / "test_manifest.csv"
    write_manifest_csv(csv_path, splits, "pan")

    records = load_manifest_csv(csv_path)
    assert len(records) == 1
    assert records[0]["source_group_id"] == "grp_001"
    assert records[0]["dataset"] == "pan"
    assert records[0]["target_split"] == "train"
    assert records[0]["sha256"] == "hash1"


def test_split_ratios_validation():
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        split_source_groups([], ratios=(0.5, 0.5, 0.5), seed=42)


@pytest.mark.skipif(not Path("data/processed/aadhaar").exists(), reason="Processed Aadhaar not generated")
def test_processed_aadhaar_isolation_and_pairing():
    proc_dir = Path("data/processed/aadhaar")
    manifest_path = Path("data/processed/manifests/aadhaar_source_groups.csv")

    assert manifest_path.exists()
    records = load_manifest_csv(manifest_path)
    assert len(records) == 2646

    # Verify no source_group_id appears across multiple target splits
    group_to_split = {}
    for r in records:
        gid = r["source_group_id"]
        tgt = r["target_split"]
        if gid in group_to_split:
            assert group_to_split[gid] == tgt, f"Group {gid} crosses splits: {group_to_split[gid]} vs {tgt}"
        else:
            group_to_split[gid] = tgt

    # Verify pairing on disk
    for split_name in ["train", "valid", "test"]:
        img_files = list((proc_dir / split_name / "images").glob("*.*"))
        lbl_files = list((proc_dir / split_name / "labels").glob("*.txt"))
        assert len(img_files) == len(lbl_files)
        img_stems = {p.stem for p in img_files}
        lbl_stems = {p.stem for p in lbl_files}
        assert img_stems == lbl_stems


@pytest.mark.skipif(not Path("data/processed/pan").exists(), reason="Processed PAN not generated")
def test_processed_pan_isolation_and_pairing():
    proc_dir = Path("data/processed/pan")
    manifest_path = Path("data/processed/manifests/pan_source_groups.csv")

    assert manifest_path.exists()
    records = load_manifest_csv(manifest_path)
    assert len(records) == 1726

    group_to_split = {}
    for r in records:
        gid = r["source_group_id"]
        tgt = r["target_split"]
        if gid in group_to_split:
            assert group_to_split[gid] == tgt, f"Group {gid} crosses splits: {group_to_split[gid]} vs {tgt}"
        else:
            group_to_split[gid] = tgt

    # Verify exact duplicate images stay in the same split
    sha_to_split = {}
    for r in records:
        sha = r["sha256"]
        tgt = r["target_split"]
        if sha in sha_to_split:
            assert sha_to_split[sha] == tgt, f"SHA {sha} crosses splits: {sha_to_split[sha]} vs {tgt}"
        else:
            sha_to_split[sha] = tgt

    for split_name in ["train", "valid", "test"]:
        img_files = list((proc_dir / split_name / "images").glob("*.*"))
        lbl_files = list((proc_dir / split_name / "labels").glob("*.txt"))
        assert len(img_files) == len(lbl_files)
        img_stems = {p.stem for p in img_files}
        lbl_stems = {p.stem for p in lbl_files}
        assert img_stems == lbl_stems


@pytest.mark.skipif(not Path("data/processed/aadhaar").exists(), reason="Processed Aadhaar not generated")
def test_aadhaar_polygons_and_class_preservation():
    proc_dir = Path("data/processed/aadhaar")
    all_labels = list(proc_dir.glob("*/**/labels/*.txt"))
    polygon_count = 0
    class_4_count = 0

    for lbl in all_labels:
        for line in lbl.read_text("utf-8-sig").splitlines():
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "4":
                class_4_count += 1
            if len(tokens) > 5:
                polygon_count += 1

    # Raw audit had 87 class-4 instances and 4 polygon lines
    assert class_4_count == 87
    assert polygon_count == 4
