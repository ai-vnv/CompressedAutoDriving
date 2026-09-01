"""One-frame detector output contracts, independent of Ultralytics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ObjectClass(str, Enum):
    STOP_SIGN = "stop_sign"
    DUCKIE = "duckie"


# This tuple is the single Version-1 detector class-order contract. Dataset
# labels, YOLO checkpoints, and runtime adapters must all agree with it.
YOLO_V1_CLASS_NAMES: tuple[str, ...] = (
    ObjectClass.STOP_SIGN.value,
    ObjectClass.DUCKIE.value,
)


def yolo_class_id(object_class: ObjectClass) -> int:
    return YOLO_V1_CLASS_NAMES.index(object_class.value)


def object_class_from_yolo_id(class_id: int) -> ObjectClass:
    if class_id < 0:
        raise ValueError(f"unsupported YOLO Version-1 class id: {class_id}")
    try:
        return ObjectClass(YOLO_V1_CLASS_NAMES[class_id])
    except (IndexError, ValueError) as error:
        raise ValueError(f"unsupported YOLO Version-1 class id: {class_id}") from error


@dataclass(frozen=True)
class ImagePoint:
    x_px: float
    y_px: float


@dataclass(frozen=True)
class BoundingBox:
    x_min_px: float
    y_min_px: float
    x_max_px: float
    y_max_px: float

    def __post_init__(self) -> None:
        coordinates = (
            self.x_min_px,
            self.y_min_px,
            self.x_max_px,
            self.y_max_px,
        )
        if not all(isfinite(value) for value in coordinates):
            raise ValueError("bounding-box coordinates must be finite")
        if self.x_min_px > self.x_max_px or self.y_min_px > self.y_max_px:
            raise ValueError("bounding-box minimums cannot exceed maximums")

    @property
    def bottom_center(self) -> ImagePoint:
        return ImagePoint(
            x_px=0.5 * (self.x_min_px + self.x_max_px),
            y_px=self.y_max_px,
        )


@dataclass(frozen=True)
class Detection:
    object_class: ObjectClass
    confidence: float
    bounding_box: BoundingBox

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detector confidence must be within [0, 1]")

    @property
    def bottom_center(self) -> ImagePoint:
        return self.bounding_box.bottom_center
