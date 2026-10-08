"""Optional training-time augmentation for synthetic OCR line crops.

Closes part of the gap between the clean synthetic crops and real detector
crops. Measured on the detector valid split, real field crops are:

- low resolution (median source height ~35-65 px, 10th percentile 15-28 px),
  so the 48 px resize upscales and softens them;
- low contrast: grey ink (5th percentile ~30-110) on a grey background
  (95th percentile ~170-205), versus near-black on near-white synthetic text;
- unevenly lit, paper-textured and JPEG-compressed;
- in condensed or bold fonts, slightly skewed, with tight or uneven margins
  and occasional underline / border fragments.

:func:`augment_line_image` takes a stored (already preprocessed) crop,
removes its padding, applies the steps below and runs
:func:`preprocess_ocr_crop` again, so the output height and padding match
the CRNN input exactly. The transcription is never changed. The clean
generator and dataset are untouched; augmentation is disabled by default.

All randomness comes from the ``np.random.Generator`` passed in, so a given
seed reproduces the same image.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, fields
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from idocr.data.augmentation.base import Augmentation, Compose
from idocr.data.preprocessing.ocr import (
    DEFAULT_HORIZONTAL_PADDING,
    DEFAULT_TARGET_HEIGHT,
    DEFAULT_VERTICAL_PADDING,
    preprocess_ocr_crop,
)
from idocr.types import ImageArray


@dataclass(frozen=True)
class OcrAugmentConfig:
    """Probabilities and ranges for each step. ``enabled=False`` = clean baseline."""

    enabled: bool = False
    # geometric
    p_hscale: float = 0.5
    hscale_range: tuple[float, float] = (0.8, 1.15)  # condensed / wide fonts
    p_affine: float = 0.4
    max_rotation_degrees: float = 1.5
    max_shear: float = 0.15
    p_margin: float = 0.5
    max_margin_pad: int = 6
    max_margin_trim: int = 2  # left/right; top/bottom trims at most 1 px
    p_edge_line: float = 0.15
    # photometric
    p_contrast: float = 0.9
    background_range: tuple[int, int] = (135, 225)
    ink_range: tuple[int, int] = (10, 125)
    min_contrast: int = 60  # background - ink, keeps characters readable
    p_illumination: float = 0.6
    max_illumination: float = 0.15
    p_texture: float = 0.5
    max_texture: float = 8.0
    p_tint: float = 0.4
    tint_range: tuple[float, float] = (0.85, 1.1)
    # degradation
    p_downsample: float = 0.6
    downsample_height_range: tuple[int, int] = (16, 40)
    p_blur: float = 0.4
    max_blur_sigma: float = 1.0  # at 48 px; scaled with the current height
    p_jpeg: float = 0.5
    jpeg_quality_range: tuple[int, int] = (30, 90)
    p_noise: float = 0.5
    max_noise_std: float = 6.0

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name.startswith("p_") and not 0.0 <= getattr(self, f.name) <= 1.0:
                raise ValueError(f"{f.name} must be in [0, 1]")
        for name in ("hscale_range", "background_range", "ink_range", "tint_range",
                     "downsample_height_range", "jpeg_quality_range"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} must be (low, high)")
        if self.hscale_range[0] <= 0 or self.downsample_height_range[0] < 8:
            raise ValueError("hscale must be positive and downsample height >= 8")
        if self.background_range[0] - self.ink_range[1] < 0 or self.min_contrast <= 0:
            raise ValueError("background_range must sit above ink_range")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "OcrAugmentConfig":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown augmentation keys: {unknown}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


def _background(arr: np.ndarray) -> int:
    return int(np.percentile(arr, 90))


def _resize(arr: np.ndarray, width: int, height: int,
            resample: Image.Resampling = Image.Resampling.BILINEAR) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((max(1, width), max(1, height)), resample))


class _Step(Augmentation):
    """Applies ``apply`` with probability ``p``."""

    def __init__(self, p: float, cfg: OcrAugmentConfig) -> None:
        self.p = p
        self.cfg = cfg

    def __call__(self, image: ImageArray, rng: np.random.Generator) -> ImageArray:
        if rng.random() < self.p:
            return self.apply(image, rng)
        return image

    def apply(self, image: ImageArray, rng: np.random.Generator) -> ImageArray:
        raise NotImplementedError


# --- geometric -------------------------------------------------------------


class HorizontalScale(_Step):
    def apply(self, image, rng):
        h, w = image.shape
        scale = rng.uniform(*self.cfg.hscale_range)
        return _resize(image, round(w * scale), h)


class SmallAffine(_Step):
    """Small rotation + horizontal shear; canvas grows so nothing is clipped."""

    def apply(self, image, rng):
        bg = _background(image)
        shear = rng.uniform(-self.cfg.max_shear, self.cfg.max_shear)
        angle = rng.uniform(-self.cfg.max_rotation_degrees, self.cfg.max_rotation_degrees)
        h, w = image.shape
        extra = int(np.ceil(abs(shear) * h))
        im = Image.fromarray(image)
        if extra:
            # x_src = x_dst + shear * (y - h/2) - offset
            offset = extra / 2
            im = im.transform(
                (w + extra, h), Image.Transform.AFFINE,
                (1, shear, -shear * h / 2 - offset, 0, 1, 0),
                resample=Image.Resampling.BILINEAR, fillcolor=bg,
            )
        im = im.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=bg)
        return np.asarray(im)


class MarginJitter(_Step):
    """Uneven margins: pad with background or trim a little (detector boxes vary)."""

    def apply(self, image, rng):
        bg = _background(image)
        trim_limits = (1, 1, self.cfg.max_margin_trim, self.cfg.max_margin_trim)  # top, bottom, left, right
        trims, pads = [0, 0, 0, 0], [0, 0, 0, 0]
        for side, limit in enumerate(trim_limits):
            if rng.random() < 0.5:
                trims[side] = int(rng.integers(0, min(limit, self.cfg.max_margin_trim) + 1))
            else:
                pads[side] = int(rng.integers(0, self.cfg.max_margin_pad + 1))
        top, bottom, left, right = trims
        h, w = image.shape
        if h - top - bottom < 8 or w - left - right < 8:
            top = bottom = left = right = 0
        out = image[top: h - bottom, left: w - right]
        out = np.pad(out, ((pads[0], pads[1]), (pads[2], pads[3])), constant_values=bg)
        return np.ascontiguousarray(out)


class EdgeLine(_Step):
    """Partial underline / border fragment along the top or bottom edge."""

    def apply(self, image, rng):
        im = Image.fromarray(image.copy())
        h, w = image.shape
        y = int(rng.integers(0, 2)) if rng.random() < 0.5 else h - 1 - int(rng.integers(0, 2))
        x0 = int(rng.integers(0, max(1, w // 3)))
        x1 = int(rng.integers(max(x0 + 1, w // 2), w + 1))
        shade = int(rng.integers(0, 110))
        ImageDraw.Draw(im).line([(x0, y), (x1, y)], fill=shade, width=1)
        return np.asarray(im)


# --- photometric -----------------------------------------------------------


class ContrastRemap(_Step):
    """Map the dark-on-light range to grey ink on a grey background."""

    def apply(self, image, rng):
        lo, hi = np.percentile(image, (1, 99))
        if hi - lo < 1:
            return image
        bg = rng.uniform(*self.cfg.background_range)
        ink_hi = min(self.cfg.ink_range[1], bg - self.cfg.min_contrast)
        ink = rng.uniform(self.cfg.ink_range[0], max(self.cfg.ink_range[0], ink_hi))
        norm = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0, 1)
        return (ink + norm * (bg - ink)).astype(np.uint8)


class Illumination(_Step):
    """Low-frequency multiplicative shading (gradient + smooth blotches)."""

    def apply(self, image, rng):
        h, w = image.shape
        amp = self.cfg.max_illumination
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        theta = rng.uniform(0, 2 * np.pi)
        ramp = (np.cos(theta) * xx / max(w, 1) + np.sin(theta) * yy / max(h, 1))
        ramp = (ramp - ramp.mean()) * rng.uniform(0, amp) * 2
        coarse = rng.uniform(-1, 1, size=(2, max(2, w // 40))).astype(np.float32)
        blotch = np.asarray(
            Image.fromarray(coarse).resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32
        ) * rng.uniform(0, amp)
        field = 1.0 + ramp + blotch
        return np.clip(image.astype(np.float32) * field, 0, 255).astype(np.uint8)


class PaperTexture(_Step):
    """Fine, slightly blurred noise added to the whole crop."""

    def apply(self, image, rng):
        noise = rng.normal(0, 1, image.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 0.8)
        noise *= rng.uniform(0, self.cfg.max_texture) / max(float(noise.std()), 1e-6)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class ColorTint(_Step):
    """Tint as RGB, then back to grayscale (the CRNN only sees luminance)."""

    def apply(self, image, rng):
        gains = rng.uniform(*self.cfg.tint_range, size=3).astype(np.float32)
        rgb = np.clip(image[..., None].astype(np.float32) * gains, 0, 255).astype(np.uint8)
        return np.asarray(Image.fromarray(rgb).convert("L"))


# --- degradation -----------------------------------------------------------


class Downsample(_Step):
    """Shrink to a low source height; preprocessing later upsamples it again."""

    def apply(self, image, rng):
        h, w = image.shape
        target = int(rng.integers(self.cfg.downsample_height_range[0],
                                  self.cfg.downsample_height_range[1] + 1))
        if target >= h:
            return image
        return _resize(image, round(w * target / h), target, Image.Resampling.BOX)


class Blur(_Step):
    def apply(self, image, rng):
        scale = image.shape[0] / DEFAULT_TARGET_HEIGHT
        sigma = rng.uniform(0.2, self.cfg.max_blur_sigma) * min(1.0, scale)
        return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(sigma)))


class JpegCompression(_Step):
    def apply(self, image, rng):
        quality = int(rng.integers(self.cfg.jpeg_quality_range[0], self.cfg.jpeg_quality_range[1] + 1))
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as im:
            return np.asarray(im.convert("L"))


class GaussianNoise(_Step):
    def apply(self, image, rng):
        noise = rng.normal(0, rng.uniform(0, self.cfg.max_noise_std), image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def build_ocr_augmentation(cfg: OcrAugmentConfig) -> Compose:
    """Steps in physical order: geometry -> appearance -> camera/compression."""
    return Compose([
        HorizontalScale(cfg.p_hscale, cfg),
        SmallAffine(cfg.p_affine, cfg),
        MarginJitter(cfg.p_margin, cfg),
        EdgeLine(cfg.p_edge_line, cfg),
        ContrastRemap(cfg.p_contrast, cfg),
        Illumination(cfg.p_illumination, cfg),
        PaperTexture(cfg.p_texture, cfg),
        ColorTint(cfg.p_tint, cfg),
        Downsample(cfg.p_downsample, cfg),
        Blur(cfg.p_blur, cfg),
        JpegCompression(cfg.p_jpeg, cfg),
        GaussianNoise(cfg.p_noise, cfg),
    ])


class OcrLineAugmenter:
    """Augments stored, already-preprocessed line crops (PIL ``L`` images)."""

    def __init__(
        self,
        config: OcrAugmentConfig,
        *,
        target_height: int = DEFAULT_TARGET_HEIGHT,
        horizontal_padding: int = DEFAULT_HORIZONTAL_PADDING,
        vertical_padding: int = DEFAULT_VERTICAL_PADDING,
    ) -> None:
        self.config = config
        self.preprocessing = dict(target_height=target_height, horizontal_padding=horizontal_padding,
                                  vertical_padding=vertical_padding)
        self.pipeline = build_ocr_augmentation(config)

    def __call__(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        if not self.config.enabled:
            return image
        v = self.preprocessing["vertical_padding"]
        h = self.preprocessing["horizontal_padding"]
        arr = np.asarray(image.convert("L"))
        inner = arr[v: arr.shape[0] - v, h: arr.shape[1] - h]
        if inner.size == 0:
            raise ValueError(f"image {image.size} is smaller than its padding")
        out = self.pipeline(np.ascontiguousarray(inner), rng)
        return preprocess_ocr_crop(Image.fromarray(out), **self.preprocessing)


def augment_line_image(image: Image.Image, config: OcrAugmentConfig, seed: int | list[int]) -> Image.Image:
    """Convenience wrapper: augment one stored crop with a fixed seed."""
    return OcrLineAugmenter(config)(image, np.random.default_rng(seed))
