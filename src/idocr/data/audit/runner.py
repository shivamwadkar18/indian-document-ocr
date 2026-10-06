"""Orchestrates a read-only audit of one dataset directory.

Flow: hash sources -> structure -> split layout -> per-split images / labels /
pairing / format-specific inspection -> dataset-level duplicate & leakage
checks -> flat issue list -> re-hash sources and fail loudly if anything changed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from idocr.data.audit.images import ImageInfo, inspect_image, summarize_images
from idocr.data.audit.integrity import compare_trees, hash_tree
from idocr.data.audit.labels import LabelInspector, preview_file, registered_inspectors
from idocr.data.audit.layout import detect_layout
from idocr.data.audit.matching import match_by_stem
from idocr.data.audit.naming import case_insensitive_collisions, roboflow_source_stem
from idocr.data.audit.report import AuditReport
from idocr.data.audit.structure import extension_of, scan_tree
from idocr.data.discovery import iter_files

DEFAULT_AUDIT_CONFIG: dict[str, Any] = {
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
    "split_name_candidates": ["train", "training", "val", "valid", "validation", "dev", "test", "testing"],
    "sample_files_per_extension": 2,
    "sample_max_lines": 5,
    "sample_max_bytes": 1024,
    "redact_previews": True,
    "min_image_side": 100,
    "max_aspect_ratio": 3.0,
    "near_duplicate_hamming": 4,
    "max_examples": 25,
    "verify_integrity": True,
}


class AuditError(RuntimeError):
    pass


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _pick_inspector(root: Path, label_files: Sequence[Path]) -> LabelInspector | None:
    for inspector in registered_inspectors():
        if inspector.can_inspect(root, label_files):
            return inspector
    return None


def _pairing_section(images: list[Path], labels: list[Path], root: Path, max_examples: int) -> dict:
    match = match_by_stem(images, labels)
    return {
        "pairs": len(match.pairs),
        "images_without_labels": [_rel(p, root) for p in match.images_without_labels],
        "labels_without_images": [_rel(p, root) for p in match.labels_without_images],
        "ambiguous_stems": match.ambiguous_stems[:max_examples],
        "ambiguous_stem_count": len(match.ambiguous_stems),
        "case_insensitive_collisions": case_insensitive_collisions(images + labels),
    }


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _duplicates_section(
    infos: Mapping[str, ImageInfo], split_of: Mapping[str, str], threshold: int, max_examples: int
) -> dict:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for rel, info in infos.items():
        if info.sha256:
            by_hash[info.sha256].append(rel)
    exact = [sorted(v) for v in by_hash.values() if len(v) > 1]
    exact_cross = [g for g in exact if len({split_of.get(p) for p in g}) > 1]

    near_cross: list[dict] = []
    items = [(rel, info.dhash, split_of.get(rel)) for rel, info in infos.items() if info.dhash is not None]
    for i, (ra, ha, sa) in enumerate(items):
        for rb, hb, sb in items[i + 1 :]:
            if sa != sb and _hamming(ha, hb) <= threshold:
                near_cross.append({"a": ra, "b": rb, "hamming": _hamming(ha, hb)})
    return {
        "exact_duplicate_groups": len(exact),
        "exact_duplicate_files": sum(map(len, exact)),
        "exact_duplicate_groups_across_splits": len(exact_cross),
        "exact_groups_across_splits": exact_cross,
        "exact_examples": exact[:max_examples],
        "near_duplicate_threshold_hamming": threshold,
        "near_duplicate_pairs_across_splits": len(near_cross),
        "near_duplicate_examples": near_cross[:max_examples],
        "near_duplicate_note": "dHash on 9x8 greyscale thumbnails; candidates only — confirm visually.",
    }


def _duplicate_label_consistency(
    infos: Mapping[str, ImageInfo], label_for: Mapping[str, Path], max_examples: int
) -> dict:
    """For byte-identical images, check whether their label files agree (line order ignored)."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for rel, info in infos.items():
        if info.sha256 and rel in label_for:
            by_hash[info.sha256].append(rel)
    groups = [g for g in by_hash.values() if len(g) > 1]
    inconsistent = []
    for g in groups:
        contents = set()
        for rel in g:
            try:
                contents.add(tuple(sorted(" ".join(l.split()) for l in label_for[rel].read_text("utf-8-sig").splitlines() if l.strip())))
            except (OSError, UnicodeDecodeError):
                contents.add(None)
        if len(contents) > 1:
            inconsistent.append(sorted(g))
    return {
        "duplicate_image_groups_with_labels": len(groups),
        "groups_with_identical_labels": len(groups) - len(inconsistent),
        "groups_with_conflicting_labels": len(inconsistent),
        "conflicting_examples": inconsistent[:max_examples],
    }


def _median(values: list[int]) -> float | None:
    return float(sorted(values)[len(values) // 2]) if values else None


def _source_naming_section(
    images: list[Path], root: Path, split_of: Mapping[str, str], infos: Mapping[str, ImageInfo], max_examples: int
) -> dict:
    sources: dict[str, list[str]] = defaultdict(list)
    roboflow_named = 0
    for p in images:
        src = roboflow_source_stem(p.stem)
        if src is not None:
            roboflow_named += 1
            sources[src].append(_rel(p, root))
    multi = {s: v for s, v in sources.items() if len(v) > 1}
    cross = {s: sorted(v) for s, v in sources.items() if len({split_of[r] for r in v}) > 1}
    split_pairs = Counter(
        " + ".join(sorted({split_of[r] for r in v})) for v in cross.values()
    )

    # Evidence that a shared source name means "same underlying picture":
    # compare dHash distances within source groups against a deterministic
    # baseline of pairs from different sources.
    hashes = {r: infos[r].dhash for v in sources.values() for r in v if infos[r].dhash is not None}
    within = [
        (hashes[a] ^ hashes[b]).bit_count()
        for v in multi.values()
        for i, a in enumerate(v)
        for b in v[i + 1 :]
        if a in hashes and b in hashes
    ]
    reps = [v[0] for v in sources.values() if v[0] in hashes]
    baseline = [(hashes[reps[i]] ^ hashes[reps[i + 1]]).bit_count() for i in range(len(reps) - 1)]
    return {
        "convention": "<original name>.rf.<32-hex>  (Roboflow export)",
        "images_matching_convention": roboflow_named,
        "images_total": len(images),
        "distinct_source_names": len(sources),
        "source_names_with_multiple_exports": len(multi),
        "images_from_multi_export_sources": sum(map(len, multi.values())),
        "max_exports_per_source": max((len(v) for v in sources.values()), default=0),
        "source_names_in_multiple_splits": len(cross),
        "images_in_cross_split_sources": sum(map(len, cross.values())),
        "cross_split_combinations": dict(split_pairs),
        "cross_split_examples": dict(list(cross.items())[:max_examples]),
        "dhash_median_within_same_source": _median(within),
        "dhash_median_between_sources": _median(baseline),
        "similarity_note": "Lower within-source distance than between-source suggests exports of one source are "
                           "variants (e.g. augmentations) of the same picture. Statistical evidence only.",
    }


def run_audit(
    dataset_dir: str | Path,
    reports_dir: str | Path | None = None,
    *,
    name: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> AuditReport:
    """Audit ``dataset_dir`` without modifying it; optionally write the JSON report.

    Raises ``AuditError`` if the dataset is missing, if ``reports_dir`` lies
    inside the dataset, or if source files changed during the audit.
    """
    root = Path(dataset_dir).resolve()
    if not root.is_dir():
        raise AuditError(f"Dataset directory does not exist: {root}")
    if reports_dir is not None:
        out = Path(reports_dir).resolve()
        if out == root or root in out.parents:
            raise AuditError(f"Refusing to write reports inside the dataset: {out}")

    cfg = {**DEFAULT_AUDIT_CONFIG, **(config or {})}
    image_exts = {e.lower() for e in cfg["image_extensions"]}
    max_ex = cfg["max_examples"]
    before = hash_tree(root) if cfg["verify_integrity"] else None

    report = AuditReport(dataset_name=name or root.name, dataset_root=str(root))
    files = list(iter_files(root))
    images = [p for p in files if extension_of(p) in image_exts]
    others = [p for p in files if extension_of(p) not in image_exts]
    issues: list[dict[str, Any]] = []

    # 1. Structure + split layout
    report.add_section("structure", scan_tree(root, cfg["split_name_candidates"]).to_dict())
    layout = detect_layout(root, cfg["split_name_candidates"])
    report.add_section(
        "layout",
        {
            "pattern": "<split>/images/<stem>.<img>  +  <split>/labels/<stem>.<label>" if layout.splits else None,
            "splits": [s.to_dict(root) for s in layout.splits],
            "unexpected_entries": layout.unexpected,
        },
    )
    issues += [{"split": None, "path": p, "problem": "unexpected_entry"} for p in layout.unexpected]

    # 2. Image inspection (every image, once)
    infos = {_rel(p, root): inspect_image(p) for p in images}
    split_of: dict[str, str] = {}
    for s in layout.splits:
        for p in files:
            if s.root in p.parents:
                split_of[_rel(p, root)] = s.name

    # Inspector is chosen from files in the detected labels/ dirs only, so stray
    # files elsewhere (READMEs, notes) don't prevent format detection.
    label_dirs = {s.labels_dir for s in layout.splits}
    label_candidates = [p for p in others if p.parent in label_dirs] if layout.splits else others
    inspector = _pick_inspector(root, label_candidates) if label_candidates else None
    report.add_section("label_format", {"inspector": inspector.name if inspector else None})

    def image_summary(paths: list[Path]) -> dict:
        return summarize_images(
            (infos[_rel(p, root)] for p in paths),
            root,
            min_side=cfg["min_image_side"],
            max_aspect=cfg["max_aspect_ratio"],
        )

    # 3. Per-split analysis
    splits_out: dict[str, Any] = {}
    groups = (
        [(s.name, s.images_dir, s.labels_dir) for s in layout.splits]
        if layout.splits
        else [("<whole dataset>", None, None)]
    )
    for split_name, images_dir, labels_dir in groups:
        if images_dir is None:
            s_imgs, s_lbls, unexpected = images, others, []
        else:
            s_imgs = [p for p in images if p.parent == images_dir]
            s_lbls = [p for p in others if p.parent == labels_dir]
            # non-images inside images/ and images inside labels/ are unexpected
            known = set(s_imgs) | set(s_lbls)
            unexpected = [_rel(p, root) for p in files if p.parent in (images_dir, labels_dir) and p not in known]
        sizes = {p.stem: (infos[_rel(p, root)].width, infos[_rel(p, root)].height)
                 for p in s_imgs if infos[_rel(p, root)].ok}
        pairing = _pairing_section(s_imgs, s_lbls, root, max_ex)
        section: dict[str, Any] = {
            "images_dir": _rel(images_dir, root) if images_dir else None,
            "labels_dir": _rel(labels_dir, root) if labels_dir else None,
            "image_files": len(s_imgs),
            "label_files": len(s_lbls),
            "total_files": len(s_imgs) + len(s_lbls) + len(unexpected),
            "image_extensions": dict(Counter(p.suffix for p in s_imgs)),
            "label_extensions": dict(Counter(p.suffix for p in s_lbls)),
            "unexpected_files": unexpected,
            "images": image_summary(s_imgs),
            "pairing": pairing,
            "annotations": inspector.inspect(root, s_lbls, sizes) if inspector and s_lbls else None,
        }
        splits_out[split_name] = section

        sn = split_name if images_dir else None
        issues += [{"split": sn, "path": p, "problem": "unexpected_file"} for p in unexpected]
        issues += [{"split": sn, "path": c["path"], "problem": "corrupted_image", "detail": c["error"]}
                   for c in section["images"]["corrupted"]]
        issues += [{"split": sn, "path": p, "problem": "image_without_label"} for p in pairing["images_without_labels"]]
        issues += [{"split": sn, "path": p, "problem": "label_without_image"} for p in pairing["labels_without_images"]]
        issues += [{"split": sn, "path": ", ".join(g), "problem": "case_insensitive_name_collision"}
                   for g in pairing["case_insensitive_collisions"]]
        rel_imgs = [_rel(p, root) for p in s_imgs]
        issues += [{"split": sn, "path": r, "problem": "exif_orientation_not_normal",
                    "detail": f"orientation={infos[r].exif_orientation}"}
                   for r in rel_imgs if infos[r].exif_orientation not in (None, 1)]
        if section["annotations"]:
            issues += [{"split": sn, **i} for i in section["annotations"]["issues"]]
            tiny = section["annotations"].get("tiny_boxes", {})
            issues += [{"split": sn, "path": t["path"], "line": t["line"], "problem": "tiny_box",
                        "detail": f"{t['w_px']}x{t['h_px']} px (< {tiny['threshold_px']} px)"}
                       for t in tiny.get("items", [])]

    report.add_section("splits", splits_out)

    # 4. Pairing rule verification
    if layout.splits:
        clean = all(
            not s["pairing"]["images_without_labels"]
            and not s["pairing"]["labels_without_images"]
            and not s["pairing"]["ambiguous_stem_count"]
            for s in splits_out.values()
        )
        report.add_section(
            "pairing_rule",
            {
                "rule": "Within each split: images/<stem>.<image ext>  <->  labels/<stem>.<label ext> "
                        "(identical, case-sensitive filename stem).",
                "status": "verified: holds for every file" if clean else "holds with exceptions (see issues)",
            },
        )
    else:
        report.add_section("pairing_rule", {"rule": "stem heuristic (unverified: no known split layout)",
                                            "status": "unverified"})

    # 5. Dataset-level duplicate / leakage checks
    duplicates = _duplicates_section(infos, split_of, cfg["near_duplicate_hamming"], max_ex)
    if layout.splits:
        labels_by_key = {(l.parent, l.stem): l for l in others}
        label_for = {
            _rel(img, root): labels_by_key[(s.labels_dir, img.stem)]
            for s in layout.splits
            for img in images
            if img.parent == s.images_dir and (s.labels_dir, img.stem) in labels_by_key
        }
        duplicates["label_consistency"] = _duplicate_label_consistency(infos, label_for, max_ex)
        issues += [{"split": None, "path": ", ".join(g), "problem": "duplicate_images_conflicting_labels"}
                   for g in duplicates["label_consistency"]["conflicting_examples"]]
    report.add_section("duplicates", duplicates)
    issues += [{"split": None, "path": ", ".join(g), "problem": "exact_duplicate_across_splits"}
               for g in duplicates["exact_groups_across_splits"]]
    if layout.splits:
        report.add_section("source_naming", _source_naming_section(images, root, split_of, infos, max_ex))

    # 6. Redacted previews (format identification aid)
    by_ext: dict[str, list[Path]] = defaultdict(list)
    for p in others:
        by_ext[extension_of(p)].append(p)
    report.add_section(
        "label_previews",
        {
            ext: {
                p_rel: preview_file(p, cfg["sample_max_lines"], cfg["sample_max_bytes"], cfg["redact_previews"])
                for p_rel, p in ((_rel(p, root), p) for p in paths[: cfg["sample_files_per_extension"]])
            }
            for ext, paths in sorted(by_ext.items())
        },
    )

    report.add_section("issue_counts", dict(Counter(i["problem"] for i in issues)))
    report.add_section("issues", issues)
    if inspector is None and others:
        report.notes.append("No registered label inspector matched; label structure needs manual inspection.")

    # 7. Integrity: the audit must not have changed anything
    if before is not None:
        integrity = compare_trees(before, hash_tree(root))
        report.add_section("source_integrity", integrity.to_dict())
        if integrity.modified:
            raise AuditError(f"SOURCE DATA MODIFIED during audit of {root}: {integrity.to_dict()}")

    if reports_dir is not None:
        report.write(reports_dir)
    return report
