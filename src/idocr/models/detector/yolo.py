"""Ultralytics YOLO field detector adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from idocr.models.detector.base import TextDetector
from idocr.types import BBox, ImageArray, TextRegion


class UltralyticsFieldDetector(TextDetector):
    """Run a trained Ultralytics YOLO detector through the project interface."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.70,
        max_det: int = 300,
    ) -> None:
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Detector checkpoint not found: {model_path}"
            )

        from ultralytics import YOLO

        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

        self.model = YOLO(str(model_path))

        raw_names = getattr(self.model, "names", {})
        self.class_names = {
            int(class_id): str(class_name)
            for class_id, class_name in raw_names.items()
        }

    def detect(self, image: ImageArray) -> list[TextRegion]:
        """Detect document fields and return pixel-coordinate regions."""

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy ndarray")

        if image.dtype != np.uint8:
            raise ValueError("image must have dtype uint8")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape H x W x 3")

        # Project-wide image convention is RGB.
        pil_image = Image.fromarray(image, mode="RGB")

        results = self.model.predict(
            source=pil_image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        regions: list[TextRegion] = []

        for box, confidence, class_id in zip(
            xyxy,
            confidences,
            class_ids,
            strict=True,
        ):
            x1, y1, x2, y2 = (float(v) for v in box)

            regions.append(
                TextRegion(
                    bbox=BBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=self.class_names.get(int(class_id)),
                )
            )

        # Deterministic downstream ordering.
        regions.sort(
            key=lambda region: region.confidence,
            reverse=True,
        )

        return regions
