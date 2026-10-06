"""Per-split visualization run, metadata index and class-identification report.

Layout under the visualization root (default ``data/reports/visualizations``)::

    <dataset>/<split>/class_<id>.jpg        per-class sheet (that class highlighted)
    <dataset>/<split>/mixed.jpg             representative mixed sample
    <dataset>/<split>/anomaly_<kind>.jpg    one sheet per anomaly kind that was found
    <dataset>/<split>/index.json            metadata for every tile (no image content)
    class_identification_report.md          rebuilt from all index.json files
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from idocr.data.visualization.dataset import LabeledImage, load_split
from idocr.data.visualization.render import contact_sheet
from idocr.data.visualization.sampling import find_anomalies, representative_sample

REPORT_FILENAME = "class_identification_report.md"
INDEX_FILENAME = "index.json"


class VisualizationError(RuntimeError):
    pass


def check_output_dir(out_dir: Path, forbidden_roots: Iterable[Path]) -> None:
    """Refuse to write inside any raw-data directory."""
    out = out_dir.resolve()
    for root in forbidden_roots:
        r = Path(root).resolve()
        if out == r or r in out.parents:
            raise VisualizationError(f"Refusing to write visualizations inside raw data: {out}")


def _remove_previous_outputs(split_dir: Path) -> None:
    """Delete only the sheets listed in a previous index.json of this split dir."""
    index = split_dir / INDEX_FILENAME
    if not index.is_file():
        return
    for sheet in json.loads(index.read_text(encoding="utf-8")).get("sheets", []):
        path = (split_dir / Path(sheet["file"]).name).resolve()
        if path.parent == split_dir.resolve() and path.suffix == ".jpg" and path.is_file():
            path.unlink()


def _save_sheet(
    split_dir: Path, filename: str, kind: str, key: str, title: str, items: Sequence[LabeledImage],
    class_ids: Sequence[int], highlight: int | None, tile_size: int, columns: int,
) -> dict[str, Any]:
    sheet, tiles = contact_sheet(
        title, items, tile_size=tile_size, columns=columns, highlight_class=highlight, class_ids=class_ids
    )
    path = split_dir / filename
    sheet.save(path, "JPEG", quality=90)
    return {
        "file": filename,
        "kind": kind,
        "key": key,
        "title": title,
        "tiles": [
            {
                "tile": t.index,
                "split": t.item.split,
                "source_filename": t.item.filename,
                "source_group": t.item.source,
                "class_ids": sorted(t.item.class_counts),
                "class_counts": {str(c): n for c, n in sorted(t.item.class_counts.items())},
                "annotation_types": t.item.annotation_types,
                "image_size": list(t.image_size),
            }
            for t in tiles
        ],
    }


def visualize_split(
    dataset: str,
    dataset_root: Path,
    split: str,
    out_root: Path,
    *,
    samples_per_class: int = 5,
    mixed_samples: int = 12,
    anomaly_samples: int | None = None,
    seed: int = 42,
    tile_size: int = 640,
    columns: int = 3,
) -> dict[str, Any]:
    """Generate all sheets for one split and write its index.json. Raw files are only read."""
    items = load_split(dataset_root, split)
    anomaly_samples = samples_per_class if anomaly_samples is None else anomaly_samples
    split_dir = out_root / dataset / split
    split_dir.mkdir(parents=True, exist_ok=True)
    _remove_previous_outputs(split_dir)

    ann_counts: Counter[int] = Counter()
    file_counts: Counter[int] = Counter()
    for i in items:
        ann_counts.update(i.class_counts)
        file_counts.update(i.class_counts.keys())
    class_ids = sorted(ann_counts)
    prefix = f"{dataset} / {split}"
    sheets = []

    for cid in class_ids:
        chosen = representative_sample(
            [i for i in items if cid in i.class_counts], samples_per_class, f"{seed}:{dataset}:{split}:class{cid}"
        )
        sheets.append(_save_sheet(split_dir, f"class_{cid}.jpg", "class", str(cid),
                                  f"{prefix} - class={cid} samples (class={cid} drawn thicker)",
                                  chosen, class_ids, cid, tile_size, columns))

    mixed = representative_sample(items, mixed_samples, f"{seed}:{dataset}:{split}:mixed")
    sheets.append(_save_sheet(split_dir, "mixed.jpg", "mixed", "mixed", f"{prefix} - mixed samples",
                              mixed, class_ids, None, tile_size, columns))

    anomalies = {}
    for anomaly in find_anomalies(items):
        entry: dict[str, Any] = {"description": anomaly.description, "found": len(anomaly.items),
                                 "visualized": 0, "sheet": None}
        if anomaly.items:
            chosen = representative_sample(anomaly.items, anomaly_samples, f"{seed}:{dataset}:{split}:{anomaly.kind}")
            fname = f"anomaly_{anomaly.kind}.jpg"
            sheets.append(_save_sheet(split_dir, fname, "anomaly", anomaly.kind,
                                      f"{prefix} - anomaly: {anomaly.kind} ({anomaly.description})",
                                      chosen, class_ids, None, tile_size, columns))
            entry.update(visualized=len(chosen), sheet=fname)
        anomalies[anomaly.kind] = entry

    index = {
        "dataset": dataset,
        "split": split,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "samples_per_class": samples_per_class,
        "images": len(items),
        "class_stats": {str(c): {"annotations": ann_counts[c], "files": file_counts[c]} for c in class_ids},
        "sheets": sheets,
        "anomalies": anomalies,
    }
    (split_dir / INDEX_FILENAME).write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _row(values: Iterable[Any]) -> str:
    return "| " + " | ".join("" if v is None else str(v) for v in values) + " |"


def render_report(indexes: Sequence[dict[str, Any]], out_root: Path) -> str:
    L = [
        "# Class identification report",
        "",
        "Local-only. Generated by `scripts/visualize_annotations.py`. Contains filenames and",
        "annotation metadata only; sheets live next to this file and are git-ignored.",
        "",
        "Class meanings are **intentionally not inferred**. Fill in *My interpretation* after",
        "inspecting the sheets, then record the result in `configs/class_mapping.yaml`.",
        "",
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for idx in indexes:
        by_dataset.setdefault(idx["dataset"], []).append(idx)

    for dataset, idxs in sorted(by_dataset.items()):
        idxs = sorted(idxs, key=lambda i: i["split"])
        L += [f"## {dataset.upper() if dataset == 'pan' else dataset.capitalize()}", "",
              f"Splits visualized: {', '.join(i['split'] for i in idxs)}", ""]

        class_ids = sorted({int(c) for i in idxs for c in i["class_stats"]})
        L += [_row(["Class ID", "Annotations", "Files with class", "Sample tiles", "Visual samples", "My interpretation"]),
              _row(["---"] * 6)]
        for cid in class_ids:
            ann = sum(i["class_stats"].get(str(cid), {}).get("annotations", 0) for i in idxs)
            files = sum(i["class_stats"].get(str(cid), {}).get("files", 0) for i in idxs)
            sheets = [(i["split"], s) for i in idxs for s in i["sheets"] if s["kind"] == "class" and s["key"] == str(cid)]
            tiles = sum(len(s["tiles"]) for _, s in sheets)
            links = "<br>".join(f"`{dataset}/{sp}/{s['file']}`" for sp, s in sheets)
            L.append(_row([cid, ann, files, tiles, links, ""]))
        L.append("")

        L += ["### Anomalies", "", _row(["Split", "Anomaly", "Description", "Files found", "Visualized", "Sheet"]),
              _row(["---"] * 6)]
        for i in idxs:
            for kind, a in i["anomalies"].items():
                sheet = f"`{dataset}/{i['split']}/{a['sheet']}`" if a["sheet"] else "Not found in this split"
                L.append(_row([i["split"], kind, a["description"], a["found"], a["visualized"], sheet]))
        L.append("")

        L += ["### Sample index", "",
              _row(["Dataset/split", "Sheet", "Tile", "Source filename", "Class IDs present",
                    "Annotation type", "Image dimensions", "Visualization output path"]),
              _row(["---"] * 8)]
        for i in idxs:
            for s in i["sheets"]:
                for t in s["tiles"]:
                    counts = ", ".join(f"{c}x{n}" if n > 1 else c for c, n in t["class_counts"].items()) or "none"
                    L.append(_row([f"{dataset}/{t['split']}", s["file"], t["tile"], t["source_filename"], counts,
                                   "+".join(t["annotation_types"]) or "-", "x".join(map(str, t["image_size"])),
                                   f"`{(out_root / dataset / i['split'] / s['file']).as_posix()}`"]))
        L.append("")
    return "\n".join(L)


def write_report(out_root: Path) -> Path:
    indexes = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_root.glob(f"*/*/{INDEX_FILENAME}"))]
    path = out_root / REPORT_FILENAME
    path.write_text(render_report(indexes, out_root), encoding="utf-8")
    return path
