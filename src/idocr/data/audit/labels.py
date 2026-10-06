"""Non-image (label) file inventory and format-specific label inspectors.

Generic behaviour (any dataset): group non-image files by extension and show
*redacted* previews of a few samples, so a human can identify the format
without the report exposing identity text.

Format-specific inspectors are added only for formats verified in real data:

* ``YoloTxtInspector`` — verified on the Aadhaar and PAN datasets (Roboflow
  YOLO export): one ``.txt`` per image, one object per line,
  ``class_id cx cy w h`` normalised to [0, 1]; some lines are Ultralytics
  segmentation polygons ``class_id x1 y1 x2 y2 ... xn yn`` (also normalised).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Redacted previews
# ---------------------------------------------------------------------------

_SAFE_TOKEN = re.compile(r"^-?(\d{1,3}|\d*\.\d+(e-?\d+)?)$")


def redact_token(token: str) -> str:
    """Keep small integers and decimals (class IDs / coordinates); mask anything else.

    Masking preserves length and character class (letters -> x/X, digits -> #)
    so the *shape* of text fields is visible without their content. Long
    digit runs (e.g. ID numbers) are always masked.
    """
    if _SAFE_TOKEN.match(token):
        return token
    return "".join("X" if c.isupper() else "x" if c.isalpha() else "#" if c.isdigit() else c for c in token)


def preview_file(path: Path, max_lines: int, max_bytes: int, redact: bool = True) -> dict[str, Any]:
    """Short read-only preview: (redacted) text lines if UTF-8, else size only."""
    with open(path, "rb") as f:
        head = f.read(max_bytes)
    size = path.stat().st_size
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return {"size_bytes": size, "encoding": "binary"}
    lines = text.splitlines()[:max_lines]
    if redact:
        lines = [" ".join(redact_token(t) for t in line.split()) for line in lines]
    return {"size_bytes": size, "encoding": "utf-8", "redacted": redact, "lines": lines, "truncated": size > len(head)}


# ---------------------------------------------------------------------------
# Inspector interface + registry
# ---------------------------------------------------------------------------


@dataclass
class LabelIssue:
    path: str  # relative to the dataset root
    problem: str
    line: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "problem": self.problem, "detail": self.detail}


class LabelInspector(ABC):
    """Format-specific label inspection. Implementations must be read-only."""

    #: Short identifier shown in reports, e.g. "yolo_txt".
    name: str = "base"

    @abstractmethod
    def can_inspect(self, dataset_root: Path, label_files: Sequence[Path]) -> bool:
        """Return True if these files are in the format this inspector understands."""

    @abstractmethod
    def inspect(
        self,
        dataset_root: Path,
        label_files: Sequence[Path],
        image_sizes: Mapping[str, tuple[int, int]] | None = None,
    ) -> dict[str, Any]:
        """Structured findings. ``image_sizes`` maps label stem -> (width, height) of its image."""


_REGISTRY: list[LabelInspector] = []


def register_inspector(inspector: LabelInspector) -> None:
    _REGISTRY.append(inspector)


def registered_inspectors() -> list[LabelInspector]:
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# YOLO txt (verified format)
# ---------------------------------------------------------------------------

#: Tolerance for floating-point rounding when checking [0, 1] bounds.
_EPS = 1e-6


@dataclass
class YoloAnnotation:
    line: int
    class_id: int
    kind: str  # "bbox" or "polygon"
    # Normalised box. For polygons: the bounding box of the points.
    cx: float
    cy: float
    w: float
    h: float
    points: list[tuple[float, float]] | None = None


@dataclass
class YoloLabelFile:
    annotations: list[YoloAnnotation] = field(default_factory=list)
    issues: list[LabelIssue] = field(default_factory=list)
    empty: bool = False
    trailing_newline: bool = False
    crlf: bool = False
    bom: bool = False
    text: str = ""


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    n = len(points)
    return abs(sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2


def parse_yolo_file(path: Path, rel: str) -> YoloLabelFile:
    """Parse one YOLO label file, collecting issues instead of raising."""
    out = YoloLabelFile()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        out.issues.append(LabelIssue(rel, "unreadable_label_file", detail=str(exc)))
        return out
    out.bom = raw.startswith(b"\xef\xbb\xbf")
    out.crlf = b"\r\n" in raw
    out.trailing_newline = raw.endswith(b"\n")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        out.issues.append(LabelIssue(rel, "encoding_error", detail=str(exc)))
        return out
    out.text = text
    if not text.strip():
        out.empty = True
        out.issues.append(LabelIssue(rel, "empty_label_file", detail="no annotations (YOLO: image with no objects)"))
        return out

    seen: dict[str, int] = {}
    for line_no, line in enumerate(text.splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        norm = " ".join(tokens)
        if norm in seen:
            out.issues.append(LabelIssue(rel, "duplicate_annotation", line_no, f"same as line {seen[norm]}"))
        seen.setdefault(norm, line_no)
        try:
            values = [float(t) for t in tokens]
        except ValueError:
            out.issues.append(LabelIssue(rel, "malformed_line", line_no, "non-numeric token"))
            continue
        if not values[0].is_integer() or values[0] < 0:
            out.issues.append(LabelIssue(rel, "invalid_class_id", line_no, tokens[0]))
            continue
        class_id, coords = int(values[0]), values[1:]

        if len(coords) == 4:
            cx, cy, w, h = coords
            ann = YoloAnnotation(line_no, class_id, "bbox", cx, cy, w, h)
            if w <= 0 or h <= 0:
                out.issues.append(LabelIssue(rel, "zero_or_negative_size", line_no, f"w={w} h={h}"))
            if any(v < -_EPS or v > 1 + _EPS for v in coords):
                out.issues.append(LabelIssue(rel, "coord_out_of_range", line_no, "value outside [0,1]"))
            elif cx - w / 2 < -_EPS or cy - h / 2 < -_EPS or cx + w / 2 > 1 + _EPS or cy + h / 2 > 1 + _EPS:
                out.issues.append(LabelIssue(rel, "bbox_outside_image", line_no, "box edge outside [0,1]"))
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            pts = list(zip(coords[0::2], coords[1::2]))
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            ann = YoloAnnotation(line_no, class_id, "polygon", (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, pts)
            if any(v < -_EPS or v > 1 + _EPS for v in coords):
                out.issues.append(LabelIssue(rel, "coord_out_of_range", line_no, "polygon point outside [0,1]"))
            if _polygon_area(pts) <= _EPS:
                out.issues.append(LabelIssue(rel, "zero_area_polygon", line_no))
        else:
            out.issues.append(LabelIssue(rel, "invalid_token_count", line_no, f"{len(tokens)} tokens"))
            continue
        out.annotations.append(ann)
    return out


def _num_stats(values: Sequence[float], digits: int = 4) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": round(min(values), digits),
        "median": round(median(values), digits),
        "mean": round(sum(values) / len(values), digits),
        "max": round(max(values), digits),
    }


class YoloTxtInspector(LabelInspector):
    """Inspector for YOLO txt labels (boxes + optional segmentation polygons)."""

    name = "yolo_txt"

    def __init__(self, tiny_box_px: int = 4) -> None:
        self.tiny_box_px = tiny_box_px

    def can_inspect(self, dataset_root: Path, label_files: Sequence[Path]) -> bool:
        txt = [p for p in label_files if p.suffix.lower() == ".txt"]
        if not txt or len(txt) != len(label_files):
            return False
        saw_line = False
        for p in txt:
            try:
                text = p.read_bytes().decode("utf-8-sig")
            except (OSError, UnicodeDecodeError):
                return False
            for line in text.splitlines():
                tokens = line.split()
                if not tokens:
                    continue
                saw_line = True
                if len(tokens) < 5:
                    return False
                try:
                    values = [float(t) for t in tokens]
                except ValueError:
                    return False
                if not values[0].is_integer():
                    return False
        return saw_line

    def inspect(
        self,
        dataset_root: Path,
        label_files: Sequence[Path],
        image_sizes: Mapping[str, tuple[int, int]] | None = None,
    ) -> dict[str, Any]:
        image_sizes = image_sizes or {}
        issues: list[LabelIssue] = []
        tokens_per_line: Counter[int] = Counter()
        kinds: Counter[str] = Counter()
        class_counts: Counter[int] = Counter()
        per_file_counts: list[int] = []
        instances_per_file: dict[int, Counter[int]] = defaultdict(Counter)
        geometry: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        polygon_points: Counter[int] = Counter()
        polygon_closed = 0
        polygon_classes: Counter[int] = Counter()
        content_groups: dict[str, list[str]] = defaultdict(list)
        fmt = Counter()
        tiny: list[dict[str, Any]] = []

        for p in label_files:
            rel = p.relative_to(dataset_root).as_posix()
            parsed = parse_yolo_file(p, rel)
            issues.extend(parsed.issues)
            fmt["files"] += 1
            fmt["empty_files"] += parsed.empty
            fmt["no_trailing_newline"] += not parsed.trailing_newline and not parsed.empty
            fmt["crlf"] += parsed.crlf
            fmt["bom"] += parsed.bom
            if parsed.text.strip():
                content_groups[" ".join(parsed.text.split())].append(rel)
            for line in parsed.text.splitlines():
                if line.split():
                    tokens_per_line[len(line.split())] += 1

            per_file_counts.append(len(parsed.annotations))
            file_classes: Counter[int] = Counter()
            size = image_sizes.get(p.stem)
            for a in parsed.annotations:
                kinds[a.kind] += 1
                class_counts[a.class_id] += 1
                file_classes[a.class_id] += 1
                g = geometry[a.class_id]
                for key in ("cx", "cy", "w", "h"):
                    g[key].append(getattr(a, key))
                if size is not None:
                    pw, ph = a.w * size[0], a.h * size[1]
                    g["w_px"].append(pw)
                    g["h_px"].append(ph)
                    if 0 < min(pw, ph) < self.tiny_box_px:
                        tiny.append({"path": rel, "line": a.line, "w_px": round(pw, 2), "h_px": round(ph, 2)})
                if a.kind == "polygon":
                    assert a.points is not None
                    polygon_points[len(a.points)] += 1
                    polygon_classes[a.class_id] += 1
                    polygon_closed += a.points[0] == a.points[-1]
            for cid, n in file_classes.items():
                instances_per_file[cid][n] += 1

        n_files = len(per_file_counts)
        per_class = {}
        for cid in sorted(class_counts):
            files_with = sum(v for k, v in instances_per_file[cid].items() if k > 0)
            dist = {str(k): v for k, v in sorted(instances_per_file[cid].items()) if k > 0}
            dist["0"] = n_files - files_with
            g = geometry[cid]
            per_class[str(cid)] = {
                "count": class_counts[cid],
                "files_with_class": files_with,
                "instances_per_file": dict(sorted(dist.items(), key=lambda kv: int(kv[0]))),
                "geometry_normalised": {k: _num_stats(g[k]) for k in ("cx", "cy", "w", "h")},
                "geometry_px": {k: _num_stats(g[k], 1) for k in ("w_px", "h_px")} if g.get("w_px") else None,
                "meaning": None,
            }

        dup_groups = [sorted(v) for v in content_groups.values() if len(v) > 1]
        return {
            "format": "yolo_txt",
            "description": (
                "One .txt per image; one object per line. 5 tokens = 'class_id cx cy w h' "
                "(box centre/size normalised by image width/height, 0-1). "
                "1 + 2N tokens (N>=3) = 'class_id x1 y1 ... xN yN' polygon, normalised 0-1."
            ),
            "transcriptions_present": False,
            "class_names_source": None,
            "class_names_note": "Meaning not determinable from annotation files alone.",
            "file_format": dict(fmt),
            "tokens_per_line": {str(k): v for k, v in sorted(tokens_per_line.items())},
            "annotation_count": sum(class_counts.values()),
            "annotation_kinds": dict(kinds),
            "class_ids": sorted(class_counts),
            "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
            "annotations_per_file": {
                **(_num_stats(per_file_counts, 2) or {}),
                "histogram": {str(k): v for k, v in sorted(Counter(per_file_counts).items())},
            },
            "per_class": per_class,
            "polygons": {
                "count": sum(polygon_points.values()),
                "points_per_polygon": {str(k): v for k, v in sorted(polygon_points.items())},
                "closed_first_equals_last": polygon_closed,
                "class_counts": {str(k): v for k, v in sorted(polygon_classes.items())},
            },
            "tiny_boxes": {"threshold_px": self.tiny_box_px, "count": len(tiny), "items": tiny},
            "identical_label_content": {"groups": len(dup_groups), "files_in_groups": sum(map(len, dup_groups))},
            "issue_counts": dict(Counter(i.problem for i in issues)),
            "issues": [i.to_dict() for i in issues],
        }


register_inspector(YoloTxtInspector())
