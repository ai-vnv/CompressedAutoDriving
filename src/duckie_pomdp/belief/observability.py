"""Predicted observability and effective detection probability (Gate F9c).

A miss is only evidence of absence where the sensor could have seen the
pedestrian. This module classifies where the *predicted* belief state falls
relative to the camera's field of view, and separates the *magnitude* of the
detection probability (which may still be estimated for a class, for
diagnostic purposes) from whether a miss in that class is *applied* as
likelihood evidence at all -- invariant I3: a sensor cannot inform you about
a region it does not observe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from numpy.typing import NDArray

from duckie_pomdp.domain.measurement import GroundPoint
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    ground_to_polar,
)


#: The single outside-domain miss policy supported by F9c v1. Defined once
#: here so the string exists in exactly one place; `EffectiveDetectionModel`
#: rejects any other value rather than silently treating an unrecognised
#: policy as "informative" -- invariant I3's guard rail needs its own guard.
PREDICTION_ONLY_POLICY = "prediction_only"


class ObservabilityClass(str, Enum):
    CENTER = "center"
    MID_FOV = "mid_fov"
    EDGE_FOV = "edge_fov"
    OUTSIDE_DOMAIN = "outside_domain"


@dataclass(frozen=True)
class PredictedObservability:
    observability_class: ObservabilityClass
    normalized_horizontal_offset: float | None
    predicted_range_m: float


class PredictedObservabilityModel:
    """Classify where the EKF's *predicted* state falls in the camera FOV.

    Uses only the predicted state and calibrated camera geometry -- never
    privileged simulator truth or the detector's own bounding box.
    """

    def __init__(self, projector: CalibratedGroundProjector, image_width_px: int) -> None:
        self._projector = projector
        self._image_width_px = image_width_px

    def classify(self, predicted_state: NDArray) -> PredictedObservability:
        x_left = float(predicted_state[0])
        y_forward = float(predicted_state[1])
        predicted_range_m = ground_to_polar(GroundPoint(x_left, y_forward)).range_m

        if y_forward <= 0.0:
            return PredictedObservability(
                ObservabilityClass.OUTSIDE_DOMAIN, None, predicted_range_m
            )

        try:
            pixel = self._projector.ground_to_pixel(GroundPoint(x_left, y_forward))
        except ValueError:
            return PredictedObservability(
                ObservabilityClass.OUTSIDE_DOMAIN, None, predicted_range_m
            )

        width = self._image_width_px
        if not 0.0 <= pixel.x_px < width:
            return PredictedObservability(
                ObservabilityClass.OUTSIDE_DOMAIN, None, predicted_range_m
            )

        # Thresholds mirror `_fov_region` in
        # experiments/validate_f9_yolo_ekf.py:62-71 -- the runtime binning
        # here and the evaluation binning there must agree, or the
        # per-class detection probabilities fitted from the evaluation will
        # be applied to the wrong frames at runtime.
        normalized = abs(pixel.x_px - 0.5 * width) / (0.5 * width)
        if normalized < 1.0 / 3.0:
            observability_class = ObservabilityClass.CENTER
        elif normalized < 2.0 / 3.0:
            observability_class = ObservabilityClass.MID_FOV
        else:
            observability_class = ObservabilityClass.EDGE_FOV

        return PredictedObservability(observability_class, normalized, predicted_range_m)


class EffectiveDetectionModel:
    """Per-observability-class detection probability, with an informativeness
    policy for misses predicted to fall outside the observable domain."""

    def __init__(
        self,
        probability_by_class: Mapping[ObservabilityClass, float],
        *,
        outside_domain_miss_policy: str,
    ) -> None:
        missing = set(ObservabilityClass) - set(probability_by_class)
        if missing:
            raise ValueError(
                "probability_by_class must supply every observability class; "
                f"missing: {sorted(cls.value for cls in missing)}"
            )
        if outside_domain_miss_policy != PREDICTION_ONLY_POLICY:
            # There is exactly one supported policy in F9c v1. Reject
            # anything else at construction time -- a typo or renamed key
            # that fell through silently would re-enable the belief-collapse
            # behaviour invariant I3 exists to eliminate.
            raise ValueError(
                "unsupported outside_domain_miss_policy: "
                f"{outside_domain_miss_policy!r}; only {PREDICTION_ONLY_POLICY!r} "
                "is supported"
            )
        self._probability_by_class = dict(probability_by_class)
        self._outside_domain_miss_policy = outside_domain_miss_policy

    def probability(self, observability: PredictedObservability) -> float:
        return self._probability_by_class[observability.observability_class]

    def miss_is_informative(self, observability: PredictedObservability) -> bool:
        if (
            observability.observability_class is ObservabilityClass.OUTSIDE_DOMAIN
            and self._outside_domain_miss_policy == PREDICTION_ONLY_POLICY
        ):
            return False
        return True
