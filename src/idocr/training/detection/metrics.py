"""Convert Ultralytics detection validation results into a plain metrics dict."""

from __future__ import annotations

from typing import Any, Mapping


def _f(x: Any) -> float:
    return round(float(x), 6)


def extract_detection_metrics(metrics: Any, names: Mapping[int, str]) -> dict[str, Any]:
    """Overall and per-class precision / recall / mAP50 / mAP50-95.

    ``metrics`` is an Ultralytics ``DetMetrics`` (``metrics.box`` holds arrays
    indexed by position in ``ap_class_index``). Classes with no validation
    targets are absent from ``ap_class_index`` and reported as ``None``.
    """
    box = metrics.box
    overall = {
        "precision": _f(box.mp),
        "recall": _f(box.mr),
        "mAP50": _f(box.map50),
        "mAP50-95": _f(box.map),
    }
    present = {int(c): i for i, c in enumerate(box.ap_class_index)}
    per_class = {}
    for cid, name in sorted(names.items()):
        i = present.get(cid)
        per_class[name] = {
            "class_id": cid,
            "precision": _f(box.p[i]) if i is not None else None,
            "recall": _f(box.r[i]) if i is not None else None,
            "mAP50": _f(box.ap50[i]) if i is not None else None,
            "mAP50-95": _f(box.ap[i]) if i is not None else None,
        }
    return {"overall": overall, "per_class": per_class}
