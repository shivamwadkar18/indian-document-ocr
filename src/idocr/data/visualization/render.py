"""Draw annotations and assemble contact sheets (in memory, Pillow only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from idocr.data.visualization.dataset import LabeledImage
from idocr.data.visualization.geometry import polygon_to_pixels, scale_for_display, yolo_box_to_pixels

# Colour-blind-friendly palette indexed by class ID (wraps for IDs >= 8).
PALETTE = [
    (230, 159, 0),    # orange
    (86, 180, 233),   # sky blue
    (0, 158, 115),    # green
    (204, 121, 167),  # pink
    (213, 94, 0),     # vermillion
    (0, 114, 178),    # blue
    (240, 228, 66),   # yellow
    (120, 120, 120),  # grey
]
BG = (255, 255, 255)
FG = (20, 20, 20)


def class_color(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def render_annotated(item: LabeledImage, max_side: int, highlight_class: int | None = None) -> Image.Image:
    """Image downscaled to ``max_side`` (aspect preserved) with every annotation drawn.

    Annotations of ``highlight_class`` get thicker outlines; all others are still shown.
    """
    with Image.open(item.image_path) as src:
        width, height = src.size
        scale = scale_for_display(width, height, max_side)
        img = src.convert("RGB")
        if scale < 1:
            img = img.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = _font(14)
    dw, dh = img.size
    for a in item.annotations:
        color = class_color(a.class_id)
        thick = 4 if highlight_class == a.class_id else 2
        if a.kind == "polygon" and a.points:
            pts = polygon_to_pixels(a.points, dw, dh)
            draw.line(pts + [pts[0]], fill=color, width=thick)
            anchor = min(pts, key=lambda p: (p[1], p[0]))
            label = f"class={a.class_id} (polygon)"
        else:
            x1, y1, x2, y2 = yolo_box_to_pixels(a.cx, a.cy, a.w, a.h, dw, dh)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thick)
            anchor = (x1, y1)
            label = f"class={a.class_id}"
        tw, th = _text_size(draw, label, font)
        lx = min(max(0, anchor[0]), max(0, dw - tw - 4))
        ly = anchor[1] - th - 4 if anchor[1] - th - 4 >= 0 else anchor[1]
        draw.rectangle([lx, ly, lx + tw + 4, ly + th + 4], fill=color)
        draw.text((lx + 2, ly + 1), label, fill=(0, 0, 0), font=font)
    return img


def _ellipsize(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = (max_chars - 1) // 2
    return text[:half] + "…" + text[-(max_chars - half - 1):]


def caption_lines(index: int, item: LabeledImage, size: tuple[int, int]) -> list[str]:
    counts = item.class_counts
    classes = ", ".join(f"{c}x{n}" if n > 1 else str(c) for c, n in sorted(counts.items())) or "none"
    kinds = "+".join(item.annotation_types) or "-"
    return [
        f"#{index}  {item.split}  {size[0]}x{size[1]}  classes: {classes}  ({kinds})",
        item.filename,
    ]


@dataclass
class Tile:
    index: int
    item: LabeledImage
    image_size: tuple[int, int]  # original (width, height)


def contact_sheet(
    title: str,
    items: Sequence[LabeledImage],
    *,
    tile_size: int = 640,
    columns: int = 3,
    highlight_class: int | None = None,
    class_ids: Sequence[int] = (),
) -> tuple[Image.Image, list[Tile]]:
    """Grid of annotated tiles with captions, a title bar and a class colour legend."""
    font, small = _font(18), _font(13)
    pad, caption_h, header_h = 10, 40, 64
    cols = max(1, min(columns, len(items)))
    rows = max(1, -(-len(items) // cols))
    sheet_w = cols * (tile_size + pad) + pad
    sheet_h = header_h + rows * (tile_size + caption_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), title, fill=FG, font=font)
    x = pad
    for cid in class_ids:
        label = f"class={cid}"
        tw, th = _text_size(draw, label, small)
        draw.rectangle([x, 36, x + tw + 8, 36 + th + 6], fill=class_color(cid))
        draw.text((x + 4, 38), label, fill=(0, 0, 0), font=small)
        x += tw + 16
    if not items:
        draw.text((pad, header_h), "No matching samples.", fill=FG, font=font)

    max_chars = tile_size // 7
    tiles = []
    for n, item in enumerate(items):
        r, c = divmod(n, cols)
        x0 = pad + c * (tile_size + pad)
        y0 = header_h + r * (tile_size + caption_h + pad)
        with Image.open(item.image_path) as src:
            size = src.size
        tile = render_annotated(item, tile_size, highlight_class)
        sheet.paste(tile, (x0 + (tile_size - tile.width) // 2, y0 + (tile_size - tile.height) // 2))
        for k, line in enumerate(caption_lines(n + 1, item, size)):
            draw.text((x0, y0 + tile_size + 4 + k * 17), _ellipsize(line, max_chars), fill=FG, font=small)
        tiles.append(Tile(n + 1, item, size))
    return sheet, tiles
