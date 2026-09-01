"""Pure bounding-box and YOLO-label rules for simulator silhouettes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from duckie_pomdp.domain.detection import BoundingBox


@dataclass(frozen=True)
class SilhouetteRules:
    minimum_visible_pixels: int
    minimum_width_px: float
    minimum_height_px: float
    maximum_border_touches: int
    minimum_truncated_height_px: float

    def __post_init__(self) -> None:
        if self.minimum_visible_pixels <= 0:
            raise ValueError("minimum_visible_pixels must be positive")
        if self.minimum_width_px <= 0.0 or self.minimum_height_px <= 0.0:
            raise ValueError("minimum bounding-box dimensions must be positive")
        if not 0 <= self.maximum_border_touches <= 4:
            raise ValueError("maximum_border_touches must be within [0, 4]")
        if self.minimum_truncated_height_px < 0.0:
            raise ValueError("minimum_truncated_height_px cannot be negative")


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x_center, self.y_center, self.width, self.height)
        if self.class_id not in (0, 1):
            raise ValueError("Version-1 YOLO class id must be 0 or 1")
        if not all(isfinite(value) for value in values):
            raise ValueError("YOLO box values must be finite")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("normalized YOLO coordinates must be within [0, 1]")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("normalized YOLO dimensions must be positive")

    @classmethod
    def from_pixels(
        cls,
        class_id: int,
        box: BoundingBox,
        image_width_px: int,
        image_height_px: int,
    ) -> YoloBox:
        if image_width_px <= 0 or image_height_px <= 0:
            raise ValueError("image dimensions must be positive")
        width = box.x_max_px - box.x_min_px
        height = box.y_max_px - box.y_min_px
        return cls(
            class_id=class_id,
            x_center=0.5 * (box.x_min_px + box.x_max_px) / image_width_px,
            y_center=0.5 * (box.y_min_px + box.y_max_px) / image_height_px,
            width=width / image_width_px,
            height=height / image_height_px,
        )

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )

    @classmethod
    def parse(cls, line: str) -> YoloBox:
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("YOLO label line must contain five fields")
        return cls(
            class_id=int(fields[0]),
            x_center=float(fields[1]),
            y_center=float(fields[2]),
            width=float(fields[3]),
            height=float(fields[4]),
        )


@dataclass(frozen=True)
class AnnotationDecision:
    accepted: bool
    reason: str
    border_touches: int
    yolo_box: YoloBox | None


def assess_silhouette(
    *,
    class_id: int,
    box: BoundingBox | None,
    visible_pixel_count: int,
    image_width_px: int,
    image_height_px: int,
    rules: SilhouetteRules,
) -> AnnotationDecision:
    if box is None or visible_pixel_count == 0:
        return AnnotationDecision(False, "not_rendered", 0, None)
    if visible_pixel_count < rules.minimum_visible_pixels:
        return AnnotationDecision(False, "too_few_visible_pixels", 0, None)

    width = box.x_max_px - box.x_min_px
    height = box.y_max_px - box.y_min_px
    border_touches = sum(
        (
            box.x_min_px <= 0.0,
            box.y_min_px <= 0.0,
            box.x_max_px >= float(image_width_px),
            box.y_max_px >= float(image_height_px),
        )
    )
    if width < rules.minimum_width_px or height < rules.minimum_height_px:
        return AnnotationDecision(False, "bbox_too_small", border_touches, None)
    if border_touches > rules.maximum_border_touches:
        return AnnotationDecision(False, "too_heavily_truncated", border_touches, None)
    if border_touches and height < rules.minimum_truncated_height_px:
        return AnnotationDecision(
            False,
            "truncated_semantic_fragment",
            border_touches,
            None,
        )

    return AnnotationDecision(
        True,
        "accepted",
        border_touches,
        YoloBox.from_pixels(class_id, box, image_width_px, image_height_px),
    )
