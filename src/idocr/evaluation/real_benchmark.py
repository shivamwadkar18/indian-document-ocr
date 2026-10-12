"""Private, manually verified real-crop OCR benchmark.

Isolates recognizer quality from detector errors: crops come from the
ground-truth boxes of the held-out detector **test** split.

Workflow (see ``scripts/real_crop_benchmark.py``)::

    create      seeded candidate queue per field -> exact crops + manifest.json
    transcribe  reviewer types each reference (twice); skip / unclear allowed
    evaluate    recognizer predictions vs. verified references -> metrics only

Benchmark directory (must be private: outside the repo or under a git-ignored
``data/`` / ``experiments/`` path)::

    manifest.json          immutable: seed, queue order, source mapping, crop hashes
    crops/<sample_id>.png  untouched crop pixels (the evaluation images)
    references.jsonl       append-only reviewer decisions (latest per sample wins)
    predictions/<run>.jsonl  private per-sample predictions of one evaluation
    reports/<run>.json     aggregate metrics only (no text, paths or images)

Privacy: functions here never print reference text, predictions or source
file names. Console-facing output uses opaque sample IDs and counts only.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from idocr.data.annotation_schema import DETECTOR_CLASSES
from idocr.data.audit.labels import parse_yolo_file
from idocr.evaluation.recognition import character_error_rate, exact_match_accuracy
from idocr.types import BBox
from idocr.utils.image import crop as clip_crop
from idocr.utils.image import load_image
from idocr.utils.paths import find_project_root

FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
REFERENCES_FILENAME = "references.jsonl"
CROPS_DIRNAME = "crops"
PREDICTIONS_DIRNAME = "predictions"
REPORTS_DIRNAME = "reports"

FIELDS: tuple[str, ...] = tuple(DETECTOR_CLASSES.values())
DEFAULT_TARGET_PER_FIELD = 40
MIN_CROP_SIDE = 2
#: Fields whose references are numeric; predictions with letters are counted.
NUMERIC_FIELDS: tuple[str, ...] = ("aadhaar_number", "date_of_birth")

VERIFIED, UNCLEAR, SKIPPED = "verified", "unclear", "skipped"
STATUSES = (VERIFIED, UNCLEAR, SKIPPED)
CMD_SKIP, CMD_UNCLEAR, CMD_QUIT = ":skip", ":unclear", ":quit"
# Repo paths that .gitignore keeps out of git (data/**, experiments/**).
_PRIVATE_REPO_DIRS = ("data", "experiments")
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class BenchmarkError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def ensure_private_dir(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Refuse benchmark storage that git could pick up.

    Paths outside the project are allowed (e.g. Google Drive); paths inside
    it must be under a git-ignored ``data/`` or ``experiments/`` directory.
    """
    path = Path(path).resolve()
    try:
        root = Path(project_root).resolve() if project_root else find_project_root()
    except FileNotFoundError:
        return path
    if path.is_relative_to(root):
        rel = path.relative_to(root).parts
        if not rel or rel[0] not in _PRIVATE_REPO_DIRS:
            raise BenchmarkError(
                f"benchmark directory {path} is inside the repository but not under a git-ignored "
                f"{'/ or '.join(_PRIVATE_REPO_DIRS)}/ directory; use private storage such as Google Drive"
            )
    return path


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    field: str
    document: str        # image stem: one source document
    image: str           # image file name inside the split images dir (private)
    label_line: int
    bbox: tuple[float, float, float, float]  # absolute pixels (x1, y1, x2, y2), unclipped


def sample_id_for(document: str, label_line: int, field: str) -> str:
    """Stable, opaque ID (no file name or text leaks into console output)."""
    digest = hashlib.sha256(f"{document}|{label_line}|{field}".encode()).hexdigest()
    return f"rc-{digest[:12]}"


def _find_image(images_dir: Path, stem: str) -> Path | None:
    matches = sorted(p for p in images_dir.glob(f"{glob.escape(stem)}.*") if p.is_file())
    return matches[0] if matches else None


def collect_candidates(images_dir: str | Path) -> tuple[list[Candidate], Counter]:
    """All ground-truth field boxes of one split, plus exclusion counts.

    Excluded (counted, never silently dropped): label files without an image,
    images with a non-default EXIF orientation (boxes would be ambiguous),
    unknown class IDs and boxes that are degenerate after clipping.
    """
    images_dir = Path(images_dir)
    labels_dir = images_dir.parent / "labels"
    candidates: list[Candidate] = []
    excluded: Counter = Counter()
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = _find_image(images_dir, label_path.stem)
        if image_path is None:
            excluded["missing_image"] += 1
            continue
        parsed = parse_yolo_file(label_path, label_path.name)
        if not parsed.annotations:
            continue
        with Image.open(image_path) as im:
            width, height = im.size
            orientation = im.getexif().get(0x0112)
        if orientation not in (None, 1):
            excluded["exif_rotated_image"] += len(parsed.annotations)
            continue
        for ann in parsed.annotations:
            field = DETECTOR_CLASSES.get(ann.class_id)
            if field is None:
                excluded["unknown_class"] += 1
                continue
            x1, y1 = (ann.cx - ann.w / 2) * width, (ann.cy - ann.h / 2) * height
            x2, y2 = (ann.cx + ann.w / 2) * width, (ann.cy + ann.h / 2) * height
            cx1, cy1 = max(0.0, x1), max(0.0, y1)
            cx2, cy2 = min(float(width), x2), min(float(height), y2)
            if cx2 - cx1 < MIN_CROP_SIDE or cy2 - cy1 < MIN_CROP_SIDE:
                excluded["degenerate_box"] += 1
                continue
            candidates.append(Candidate(
                sample_id=sample_id_for(label_path.stem, ann.line, field),
                field=field,
                document=label_path.stem,
                image=image_path.name,
                label_line=ann.line,
                bbox=(x1, y1, x2, y2),
            ))
    return candidates, excluded


def order_candidates(candidates: Iterable[Candidate], seed: int) -> dict[str, list[Candidate]]:
    """Seeded review queue per field, distinct documents first.

    Documents are shuffled; round 1 takes one box from every document, round 2
    a second box from documents with several, and so on. Selection is random
    (no size or quality filter), so hard crops are not avoided.
    """
    by_field: dict[str, dict[str, list[Candidate]]] = {f: defaultdict(list) for f in FIELDS}
    for c in sorted(candidates, key=lambda c: (c.field, c.document, c.label_line)):
        by_field[c.field][c.document].append(c)

    queues: dict[str, list[Candidate]] = {}
    for field in FIELDS:
        rng = random.Random(f"{seed}:{field}")
        docs = sorted(by_field[field])
        rng.shuffle(docs)
        per_doc = {}
        for d in docs:
            boxes = list(by_field[field][d])
            rng.shuffle(boxes)
            per_doc[d] = boxes
        queue: list[Candidate] = []
        depth = max((len(b) for b in per_doc.values()), default=0)
        for r in range(depth):
            queue += [per_doc[d][r] for d in docs if r < len(per_doc[d])]
        queues[field] = queue
    return queues


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def create_benchmark(
    images_dir: str | Path,
    benchmark_dir: str | Path,
    *,
    name: str,
    seed: int,
    target_per_field: int = DEFAULT_TARGET_PER_FIELD,
    source_description: str = "",
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the fixed candidate queues and export every queued crop.

    All candidates (not only the first ``target_per_field``) are exported so
    replacements for skipped crops come from the same fixed queue and the
    benchmark no longer needs the source dataset afterwards.

    Raises:
        BenchmarkError: if a manifest already exists (benchmarks are immutable)
            or the directory is not private.
    """
    if not _RUN_NAME_RE.match(name):
        raise BenchmarkError(f"invalid benchmark name {name!r}")
    bdir = ensure_private_dir(benchmark_dir, project_root)
    if (bdir / MANIFEST_FILENAME).exists():
        raise BenchmarkError(f"{bdir / MANIFEST_FILENAME} already exists; benchmarks are never overwritten")
    images_dir = Path(images_dir)
    candidates, excluded = collect_candidates(images_dir)
    queues = order_candidates(candidates, seed)

    crops_dir = bdir / CROPS_DIRNAME
    crops_dir.mkdir(parents=True, exist_ok=True)
    samples: dict[str, Any] = {}
    images_cache: dict[str, np.ndarray] = {}
    for field in FIELDS:
        for c in queues[field]:
            if c.image not in images_cache:
                images_cache.clear()  # one source image in memory at a time
                images_cache[c.image] = load_image(images_dir / c.image)
            image = images_cache[c.image]
            patch = clip_crop(image, BBox(*c.bbox))
            h, w = image.shape[:2]
            # Pixel window clip_crop used (floor start, ceil end, clipped).
            window = [max(0, min(w, int(np.floor(c.bbox[0])))), max(0, min(h, int(np.floor(c.bbox[1])))),
                      max(0, min(w, int(np.ceil(c.bbox[2])))), max(0, min(h, int(np.ceil(c.bbox[3]))))]
            crop_path = crops_dir / f"{c.sample_id}.png"
            Image.fromarray(np.ascontiguousarray(patch)).save(crop_path, format="PNG")
            samples[c.sample_id] = {
                "field": c.field,
                "document": c.document,
                "image": c.image,
                "label_line": c.label_line,
                "bbox_px": list(c.bbox),
                "crop_window_px": window,
                "crop": f"{CROPS_DIRNAME}/{crop_path.name}",
                "crop_size": [int(patch.shape[1]), int(patch.shape[0])],
                "crop_sha256": _sha256_file(crop_path),
            }

    manifest = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "created_at": _now(),
        "seed": seed,
        "split": "test",
        "box_source": "ground_truth",
        "source": {"images_dir": str(images_dir), "description": source_description},
        "target_per_field": target_per_field,
        "queues": {f: [c.sample_id for c in queues[f]] for f in FIELDS},
        "available_documents": {f: len({c.document for c in queues[f]}) for f in FIELDS},
        "excluded": dict(excluded),
        "samples": samples,
    }
    _atomic_write_json(bdir / MANIFEST_FILENAME, manifest)
    return manifest


def load_manifest(benchmark_dir: str | Path) -> dict[str, Any]:
    path = Path(benchmark_dir) / MANIFEST_FILENAME
    if not path.is_file():
        raise BenchmarkError(f"no benchmark manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise BenchmarkError(f"unsupported manifest format {manifest.get('format_version')}")
    return manifest


def manifest_sha256(benchmark_dir: str | Path) -> str:
    return _sha256_file(Path(benchmark_dir) / MANIFEST_FILENAME)


def load_crop(benchmark_dir: str | Path, manifest: Mapping[str, Any], sample_id: str,
              verify: bool = True) -> Image.Image:
    """The exact evaluation crop (RGB), optionally hash-checked."""
    meta = manifest["samples"][sample_id]
    path = Path(benchmark_dir) / meta["crop"]
    if verify and _sha256_file(path) != meta["crop_sha256"]:
        raise BenchmarkError(f"crop {sample_id} does not match its manifest hash")
    with Image.open(path) as im:
        return im.convert("RGB")


# ---------------------------------------------------------------------------
# References (append-only, resumable)
# ---------------------------------------------------------------------------


def validate_reference(text: str) -> str | None:
    """Reason the typed reference is rejected, or None. Never rewrites it."""
    if not text:
        return "empty"
    if text != text.strip():
        return "leading or trailing whitespace"
    if any(ch in text for ch in "\r\n\t"):
        return "contains a line break or tab"
    if text.startswith(":"):
        return "starts with ':' (reserved for commands)"
    return None


class ReferenceStore:
    """Append-only JSONL of reviewer decisions; the latest record per sample wins.

    Every write is flushed and fsync'd, so progress survives a runtime restart.
    """

    def __init__(self, benchmark_dir: str | Path) -> None:
        self.path = Path(benchmark_dir) / REFERENCES_FILENAME

    def records(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        if not self.path.is_file():
            return latest
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line torn by a crash mid-write is ignored
                if rec.get("status") in STATUSES and "sample_id" in rec:
                    latest[rec["sample_id"]] = rec
        return latest

    def record(self, sample_id: str, status: str, text: str | None = None) -> None:
        if status not in STATUSES:
            raise BenchmarkError(f"unknown status {status!r}")
        if status == VERIFIED:
            problem = validate_reference(text or "")
            if problem:
                raise BenchmarkError(f"invalid reference for {sample_id}: {problem}")
        elif text is not None:
            raise BenchmarkError("only verified samples carry text")
        rec = {"sample_id": sample_id, "status": status, "recorded_at": _now()}
        if status == VERIFIED:
            rec["text"] = text
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def progress(manifest: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-field counts (no text). ``exhausted``: target unreachable with this queue."""
    out = {}
    target = manifest["target_per_field"]
    for field in FIELDS:
        statuses = Counter(records[s]["status"] for s in manifest["queues"][field] if s in records)
        pending = sum(1 for s in manifest["queues"][field] if s not in records)
        out[field] = {
            VERIFIED: statuses[VERIFIED], UNCLEAR: statuses[UNCLEAR], SKIPPED: statuses[SKIPPED],
            "pending": pending, "target": target,
            "complete": statuses[VERIFIED] >= target,
            "exhausted": statuses[VERIFIED] < target and pending == 0,
        }
    return out


def next_sample(manifest: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Next undecided sample of the least-verified incomplete field (queue order)."""
    prog = progress(manifest, records)
    open_fields = [f for f in FIELDS if not prog[f]["complete"] and prog[f]["pending"]]
    if not open_fields:
        return None
    field = min(open_fields, key=lambda f: (prog[f][VERIFIED], FIELDS.index(f)))
    return next(s for s in manifest["queues"][field] if s not in records)


def display_copy(image: Image.Image, height: int = 160, max_width: int = 1600) -> Image.Image:
    """Enlarged copy for viewing only; the stored crop is never changed."""
    scale = min(height / image.height, max_width / image.width)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def transcribe(
    benchmark_dir: str | Path,
    *,
    ask: Callable[[str], str],
    show: Callable[[Image.Image, str], None],
    say: Callable[[str], None] = print,
    double_entry: bool = True,
    max_items: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Interactive reference collection; resumes where the last session stopped.

    ``show(display_image, caption)`` gets an enlarged copy and a caption with
    only the opaque sample ID, field and progress. ``ask(prompt)`` returns
    the reviewer's input. No model is loaded and no prediction is shown.
    Commands: ``:skip``, ``:unclear``, ``:quit``. With ``double_entry`` the
    reference must be typed twice identically (it is never echoed back).
    """
    bdir = Path(benchmark_dir)
    manifest = load_manifest(bdir)
    store = ReferenceStore(bdir)
    done = 0
    while max_items is None or done < max_items:
        records = store.records()
        sample_id = next_sample(manifest, records)
        if sample_id is None:
            break
        field = manifest["samples"][sample_id]["field"]
        prog = progress(manifest, records)[field]
        show(display_copy(load_crop(bdir, manifest, sample_id)),
             f"{sample_id} | field: {field} | verified {prog[VERIFIED]}/{prog['target']}")
        while True:
            text = ask(f"Type the {field} exactly as shown ({CMD_SKIP} / {CMD_UNCLEAR} / {CMD_QUIT}): ")
            if text == CMD_QUIT:
                say("Stopped; progress is saved.")
                return progress(manifest, store.records())
            if text in (CMD_SKIP, CMD_UNCLEAR):
                store.record(sample_id, SKIPPED if text == CMD_SKIP else UNCLEAR)
                break
            problem = validate_reference(text)
            if problem:
                say(f"Not saved ({problem}); type it again.")
                continue
            if double_entry and ask("Type it again to confirm: ") != text:
                say("The two entries differ; not saved. Try again.")
                continue
            store.record(sample_id, VERIFIED, text)
            break
        done += 1
    final = progress(manifest, store.records())
    for field, p in final.items():
        if p["exhausted"]:
            say(f"{field}: only {p[VERIFIED]} verified and no candidates left (target {p['target']}).")
    return final


def colab_transcribe(benchmark_dir: str | Path, **kwargs: Any) -> dict[str, dict[str, Any]]:
    """``transcribe`` wired to IPython display (Colab / Jupyter).

    Input uses ``getpass`` so typed references are not echoed into the
    notebook output (plain ``input()`` would be saved with the notebook);
    double entry catches typos. The output is cleared when the session ends.
    """
    from getpass import getpass

    from IPython.display import clear_output, display

    def show(image: Image.Image, caption: str) -> None:
        clear_output(wait=True)
        print(caption)
        display(image)

    try:
        return transcribe(benchmark_dir, ask=getpass, show=show, **kwargs)
    finally:
        clear_output()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def alpha_rate(predictions: Sequence[str]) -> float | None:
    """Share of predictions containing any alphabetic character (None if empty)."""
    if not predictions:
        return None
    return sum(any(ch.isalpha() for ch in p) for p in predictions) / len(predictions)


def text_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    """Raw (case-sensitive) and case-insensitive CER / exact match.

    CER is corpus-level (total edits / total reference characters, from
    ``idocr.evaluation.recognition``). Case-insensitive variants compare
    ``str.casefold()`` of both sides; nothing else is normalized.
    """
    if not references:
        return {"count": 0, "cer": None, "exact_match": None, "cer_ci": None, "exact_match_ci": None}
    folded_p = [p.casefold() for p in predictions]
    folded_r = [r.casefold() for r in references]
    return {
        "count": len(references),
        "cer": character_error_rate(predictions, references),
        "exact_match": exact_match_accuracy(predictions, references),
        "cer_ci": character_error_rate(folded_p, folded_r),
        "exact_match_ci": exact_match_accuracy(folded_p, folded_r),
    }


def aggregate_report(rows: Sequence[tuple[str, str, str]], vocabulary: str | None = None) -> dict[str, Any]:
    """Metrics from ``(field, reference, prediction)`` rows. Contains no text."""
    per_field: dict[str, Any] = {}
    for field in FIELDS:
        refs = [r for f, r, _ in rows if f == field]
        preds = [p for f, _, p in rows if f == field]
        entry = text_metrics(preds, refs)
        if field in NUMERIC_FIELDS:
            entry["alpha_prediction_rate"] = alpha_rate(preds)
        if vocabulary is not None:
            entry["references_outside_vocabulary"] = sum(any(ch not in vocabulary for ch in r) for r in refs)
        per_field[field] = entry
    overall = text_metrics([p for _, _, p in rows], [r for _, r, _ in rows])
    return {"overall": overall, "per_field": per_field}


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def evaluate_benchmark(
    benchmark_dir: str | Path,
    recognizer: Any,
    *,
    run_name: str,
    model_info: Mapping[str, Any] | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Score ``recognizer`` (a ``TextRecognizer``) on every verified reference.

    Predictions are scored exactly as returned by ``recognize_batch``. They are
    saved privately to ``predictions/<run_name>.jsonl``; the aggregate report
    (metrics only) is saved to ``reports/<run_name>.json`` and returned.
    """
    if not _RUN_NAME_RE.match(run_name):
        raise BenchmarkError(f"invalid run name {run_name!r}")
    bdir = Path(benchmark_dir)
    manifest = load_manifest(bdir)
    report_path = bdir / REPORTS_DIRNAME / f"{run_name}.json"
    if report_path.exists():
        raise BenchmarkError(f"report {report_path.name} already exists; choose a new run name")
    records = ReferenceStore(bdir).records()
    verified = [s for f in FIELDS for s in manifest["queues"][f]
                if records.get(s, {}).get("status") == VERIFIED]
    if not verified:
        raise BenchmarkError("no verified references yet")

    predictions: list[tuple[str, float]] = []
    for start in range(0, len(verified), batch_size):
        ids = verified[start:start + batch_size]
        crops = [np.asarray(load_crop(bdir, manifest, s)) for s in ids]
        predictions += [(r.text, r.confidence) for r in recognizer.recognize_batch(crops)]

    pred_dir = bdir / PREDICTIONS_DIRNAME
    pred_dir.mkdir(exist_ok=True)
    with (pred_dir / f"{run_name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for sample_id, (text, confidence) in zip(verified, predictions, strict=True):
            f.write(json.dumps({"sample_id": sample_id, "prediction": text, "confidence": confidence},
                               ensure_ascii=False) + "\n")

    rows = [(manifest["samples"][s]["field"], records[s]["text"], p)
            for s, (p, _) in zip(verified, predictions, strict=True)]
    vocab = getattr(getattr(recognizer, "vocab", None), "alphabet", None)
    report = {
        "benchmark": manifest["name"],
        "manifest_sha256": manifest_sha256(bdir),
        "run_name": run_name,
        "evaluated_at": _now(),
        "box_source": manifest["box_source"],
        "model": dict(model_info or {}),
        "review_status": {f: {k: v for k, v in p.items() if k in (*STATUSES, "pending", "target")}
                          for f, p in progress(manifest, records).items()},
        **aggregate_report(rows, vocab),
    }
    report_path.parent.mkdir(exist_ok=True)
    _atomic_write_json(report_path, report)
    return report


def format_report(report: Mapping[str, Any]) -> str:
    """Console table of an aggregate report (metrics only)."""
    def pct(v: float | None) -> str:
        return "   -  " if v is None else f"{100 * v:6.2f}"

    lines = [f"benchmark {report['benchmark']} | run {report['run_name']} | boxes {report['box_source']}",
             f"{'field':15s} {'n':>4s} {'CER%':>7s} {'EM%':>7s} {'CER%ci':>7s} {'EM%ci':>7s} {'alpha%':>7s}"]
    for name, m in [*report["per_field"].items(), ("overall", report["overall"])]:
        lines.append(f"{name:15s} {m['count']:4d} {pct(m['cer'])} {pct(m['exact_match'])} "
                     f"{pct(m['cer_ci'])} {pct(m['exact_match_ci'])} {pct(m.get('alpha_prediction_rate'))}")
    return "\n".join(lines)
