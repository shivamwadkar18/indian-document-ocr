"""CLI script to generate detector-ready dataset and validation report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from idocr.data.annotation_schema import DETECTOR_CLASSES
from idocr.data.audit.integrity import hash_tree
from idocr.data.detection.builder import DetectorDatasetBuilder
from idocr.data.detection.validation import DetectorDatasetValidator


def generate_detector_report(
    builder_stats: dict,
    validation_res: dict,
    output_path: Path,
) -> None:
    """Generate comprehensive Markdown report for detector dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    status_badge = "**VALID**" if validation_res["is_valid"] else "**INVALID / FAILED**"

    lines = [
        "# Common Field-Detector Dataset Generation Report",
        "",
        f"**Status**: {status_badge}  ",
        f"**Total Images**: {validation_res['total_images']}  ",
        f"**Total Detector Targets**: {validation_res['total_targets']}  ",
        f"**Excluded Aadhaar Class 4 Targets**: {builder_stats.get('excluded_class_4_count', 0)}  ",
        f"**Empty-Label Images**: {validation_res.get('empty_labels_by_split', {})}  ",
        "",
        "---",
        "",
        "## 1. Dataset Split Accounting",
        "",
        "| Split | Images | Labels | Target Count | Empty Labels |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for split in ("train", "valid", "test"):
        imgs = validation_res["images_by_split"].get(split, 0)
        lbls = validation_res["labels_by_split"].get(split, 0)
        empty = validation_res["empty_labels_by_split"].get(split, 0)
        targets = builder_stats.get("targets_by_split", {}).get(split, 0)
        lines.append(f"| `{split}` | {imgs:,} | {lbls:,} | {targets:,} | {empty} |")

    lines.extend([
        "",
        f"**Images by Source Dataset**:",
        f"- Aadhaar: {builder_stats.get('images_by_dataset', {}).get('aadhaar', 0):,} images",
        f"- PAN: {builder_stats.get('images_by_dataset', {}).get('pan', 0):,} images",
        "",
        "---",
        "",
        "## 2. Target Distribution by Detector Class",
        "",
        "| Class ID | Field Name | Description / Document | Target Count |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for cid, name in sorted(DETECTOR_CLASSES.items()):
        cnt = validation_res["targets_by_class"].get(f"{cid} ({name})", 0)
        if cnt == 0:
            cnt = builder_stats.get("targets_by_class", {}).get(cid, 0)
        desc = "Aadhaar & PAN"
        if cid == 2 or cid == 3:
            desc = "Aadhaar only"
        elif cid == 4 or cid == 5:
            desc = "PAN only"
        lines.append(f"| `{cid}` | `{name}` | {desc} | {cnt:,} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Class 4 Exclusion & Handling",
        "",
        "- **Original Aadhaar Class 4**: Excluded completely from detector training targets.",
        f"- **Class 4 Excluded Instances**: {builder_stats.get('excluded_class_4_count', 0)}",
        "- **Images with Only Class 4**: Preserved as valid background images with an empty label file (`.txt` containing 0 targets).",
        "",
        "---",
        "",
        "## 4. Integrity and Leakage Validation",
        "",
        f"- **1:1 Image/Label Pairing Passed**: {'YES' if validation_res['pairing_passed'] else 'NO'}",
        f"- **Valid Detector Class IDs (0-5)**: {'YES' if validation_res['classes_valid'] else 'NO'}",
        f"- **Valid Normalized Geometry ([0, 1])**: {'YES' if validation_res['geometry_valid'] else 'NO'}",
        f"- **Zero Cross-Split Leakage**: {'YES' if validation_res['split_leakage_free'] else 'NO'}",
        f"- **Source Dataset Immutability Passed**: {'YES' if validation_res['source_integrity_passed'] else 'NO'}",
        "",
    ])

    if validation_res["errors"]:
        lines.extend([
            "### Validation Errors:",
            "",
        ])
        for err in validation_res["errors"]:
            lines.append(f"- ❌ {err}")
        lines.append("")

    if validation_res["warnings"]:
        lines.extend([
            "### Validation Warnings:",
            "",
        ])
        for warn in validation_res["warnings"]:
            lines.append(f"- ⚠️ {warn}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Detector dataset report written to {output_path}")


def _snapshot_tree_stats(root: Path) -> dict[str, tuple[int, int]]:
    """Fast snapshot of (size, mtime_ns) for all files under root."""
    stats = {}
    if not root.exists():
        return stats
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            stats[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate unified detector-ready dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detection_classes.yaml"),
        help="Path to detection classes config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/detector"),
        help="Target detector dataset directory.",
    )
    parser.add_argument(
        "--aadhaar-dir",
        type=Path,
        default=Path("data/processed/aadhaar"),
        help="Path to processed Aadhaar dataset.",
    )
    parser.add_argument(
        "--pan-dir",
        type=Path,
        default=Path("data/processed/pan"),
        help="Path to processed PAN dataset.",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/processed/manifests"),
        help="Path to split manifests directory.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw datasets directory.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/reports/detector_dataset_report.md"),
        help="Path to output markdown report.",
    )
    args = parser.parse_args()

    print("================ GENERATING DETECTOR DATASET ================", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Config: {args.config}", flush=True)

    # Check raw and processed source integrity before build
    print("Snapshotting raw and processed source integrity before build...", flush=True)
    raw_stats_before = _snapshot_tree_stats(args.raw_dir)
    proc_aadhaar_before = _snapshot_tree_stats(args.aadhaar_dir)
    proc_pan_before = _snapshot_tree_stats(args.pan_dir)
    print(f"  Snapshotted {len(raw_stats_before)} raw, {len(proc_aadhaar_before)} Aadhaar, {len(proc_pan_before)} PAN files.", flush=True)

    builder = DetectorDatasetBuilder(
        output_dir=args.output_dir,
        source_mapping_config=args.config,
    )

    print("Building detector dataset from Aadhaar and PAN splits...", flush=True)
    stats = builder.build_dataset(
        aadhaar_processed_dir=args.aadhaar_dir,
        pan_processed_dir=args.pan_dir,
    )

    print("\nDataset Build Complete:", flush=True)
    print(f"  Total Images: {stats.total_images}", flush=True)
    print(f"  Total Targets: {stats.total_targets}", flush=True)
    print(f"  Images by Split: {stats.images_by_split}", flush=True)
    print(f"  Targets by Class: {stats.targets_by_class}", flush=True)
    print(f"  Excluded Class 4: {stats.excluded_class_4_count}", flush=True)
    print(f"  Empty Label Images: {stats.empty_label_images}", flush=True)

    # Validate
    print("\nValidating generated detector dataset...", flush=True)
    validator = DetectorDatasetValidator(
        dataset_dir=args.output_dir,
        manifests_dir=args.manifests_dir,
        raw_dir=args.raw_dir,
        processed_dir=Path("data/processed"),
    )
    val_res = validator.validate()

    # Check immutability after
    raw_stats_after = _snapshot_tree_stats(args.raw_dir)
    proc_aadhaar_after = _snapshot_tree_stats(args.aadhaar_dir)
    proc_pan_after = _snapshot_tree_stats(args.pan_dir)

    if raw_stats_before != raw_stats_after:
        val_res.add_error("FATAL: Raw datasets were modified during detector dataset generation!")
    if proc_aadhaar_before != proc_aadhaar_after:
        val_res.add_error("FATAL: Processed Aadhaar dataset was modified during detector dataset generation!")
    if proc_pan_before != proc_pan_after:
        val_res.add_error("FATAL: Processed PAN dataset was modified during detector dataset generation!")

    # Write report
    generate_detector_report(
        builder_stats=stats.to_dict(),
        validation_res=val_res.to_dict(),
        output_path=args.report_path,
    )

    if not val_res.is_valid:
        print("\n[FAILED] Validation Failed with errors:", flush=True)
        for err in val_res.errors:
            print(f"  - {err}", flush=True)
        return 1

    print("\n[SUCCESS] Detector dataset generation and validation passed successfully!", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
