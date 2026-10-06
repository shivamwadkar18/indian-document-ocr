"""Licence layout templates.

A layout says where each field is drawn and how. Layouts are authored by us
(fictional designs inspired by the general structure of DL cards), never traced
from real licence scans.

TODO: define template file format (YAML) and author multiple layouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FieldSlot:
    field_name: str
    x: float  # relative position, 0..1
    y: float
    font_size: int | None = None
    label_text: str | None = None  # printed caption, e.g. "Name"


@dataclass
class Layout:
    name: str
    width: int
    height: int
    background: str | None = None  # path to a self-made background, if any
    slots: list[FieldSlot] = field(default_factory=list)
