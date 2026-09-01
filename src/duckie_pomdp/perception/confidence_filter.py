"""Deterministic class-specific filtering at the detector adapter boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.domain.detection import Detection, ObjectClass
from duckie_pomdp.ports.detector import ObjectDetector


class ClassConfidenceFilter:
    """Filter detector outputs without changing model weights or image input."""

    def __init__(
        self,
        detector: ObjectDetector,
        minimum_confidence: Mapping[ObjectClass, float],
    ) -> None:
        thresholds = dict(minimum_confidence)
        if any(
            not isfinite(value) or not 0.0 <= value <= 1.0
            for value in thresholds.values()
        ):
            raise ValueError("class confidence thresholds must be within [0, 1]")
        self._detector = detector
        self._minimum_confidence = thresholds

    def detect(self, rgb: NDArray[np.uint8]) -> Sequence[Detection]:
        return tuple(
            detection
            for detection in self._detector.detect(rgb)
            if detection.confidence
            >= self._minimum_confidence.get(detection.object_class, 0.0)
        )


class DuckieImageDomainFilter:
    """Reject Duckie boxes outside the calibrated Version-1 image domain.

    The rule consumes only detector output.  It is intentionally separate from
    the EKF and never receives simulator state, elapsed scenario time, or an
    object identity.
    """

    def __init__(self, detector: ObjectDetector, *, maximum_bottom_y_px: float) -> None:
        if not isfinite(maximum_bottom_y_px) or maximum_bottom_y_px <= 0.0:
            raise ValueError("maximum Duckie bottom coordinate must be positive")
        self._detector = detector
        self._maximum_bottom_y_px = float(maximum_bottom_y_px)

    def detect(self, rgb: NDArray[np.uint8]) -> Sequence[Detection]:
        return tuple(
            detection
            for detection in self._detector.detect(rgb)
            if detection.object_class is not ObjectClass.DUCKIE
            or detection.bounding_box.y_max_px <= self._maximum_bottom_y_px
        )
