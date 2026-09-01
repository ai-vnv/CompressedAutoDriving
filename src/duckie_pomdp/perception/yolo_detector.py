"""Ultralytics adapter isolated behind the image-only detector port."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from duckie_pomdp.domain.detection import (
    YOLO_V1_CLASS_NAMES,
    BoundingBox,
    Detection,
    object_class_from_yolo_id,
)


def normalize_class_names(names: Mapping[int, str] | Sequence[str]) -> tuple[str, ...]:
    """Return checkpoint names in class-id order and reject sparse mappings."""

    if isinstance(names, Mapping):
        normalized = {int(key): str(value) for key, value in names.items()}
        expected_ids = list(range(len(normalized)))
        if sorted(normalized) != expected_ids:
            raise ValueError("YOLO class mapping must be dense and zero-based")
        return tuple(normalized[index] for index in expected_ids)
    return tuple(str(value) for value in names)


def validate_v1_class_mapping(names: Mapping[int, str] | Sequence[str]) -> tuple[str, ...]:
    normalized = normalize_class_names(names)
    if normalized != YOLO_V1_CLASS_NAMES:
        raise ValueError(
            "checkpoint class mapping does not match Version 1: "
            f"expected {YOLO_V1_CLASS_NAMES}, received {normalized}"
        )
    return normalized


def rgb_to_pil(rgb: NDArray[np.uint8]) -> Image.Image:
    """Validate simulator RGB and preserve its channel order for Ultralytics."""

    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        raise ValueError("YOLO input must be uint8 RGB")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("YOLO input must have shape (height, width, 3)")
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YoloObjectDetector:
    """Two-class detector whose public inference input is front RGB only."""

    def __init__(
        self,
        weights: str | Path,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        image_size: int = 480,
        device: str | int = 0,
        max_detections: int = 300,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be in [0, 1]")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("IoU threshold must be in [0, 1]")
        if image_size <= 0:
            raise ValueError("image size must be positive")
        if max_detections <= 0:
            raise ValueError("max detections must be positive")
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(weights))
        validate_v1_class_mapping(model.names)
        self._model = model
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._image_size = image_size
        self._device = device
        self._max_detections = max_detections

    def detect(self, rgb: NDArray[np.uint8]) -> Sequence[Detection]:
        image = rgb_to_pil(rgb)
        results = self._model.predict(
            source=image,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            imgsz=self._image_size,
            device=self._device,
            max_det=self._max_detections,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = _as_numpy(boxes.xyxy).reshape(-1, 4)
            confidence = _as_numpy(boxes.conf).reshape(-1)
            class_ids = _as_numpy(boxes.cls).reshape(-1)
            if not (len(xyxy) == len(confidence) == len(class_ids)):
                raise RuntimeError("Ultralytics returned inconsistent box arrays")
            for coordinates, score, class_id_value in zip(
                xyxy, confidence, class_ids, strict=True
            ):
                class_id = int(class_id_value)
                detections.append(
                    Detection(
                        object_class=object_class_from_yolo_id(class_id),
                        confidence=float(score),
                        bounding_box=BoundingBox(
                            x_min_px=float(coordinates[0]),
                            y_min_px=float(coordinates[1]),
                            x_max_px=float(coordinates[2]),
                            y_max_px=float(coordinates[3]),
                        ),
                    )
                )
        return tuple(detections)
