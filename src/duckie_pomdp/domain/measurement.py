"""Metric object measurements produced after ground-plane projection."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .detection import ObjectClass
from .observation import EgoObservation, RoadMeasurement


@dataclass(frozen=True)
class GroundPoint:
    """Point in the ego frame: left-positive x and forward-positive y."""

    x_left_m: float
    y_forward_m: float


@dataclass(frozen=True)
class LaneMeasurement:
    """One-frame camera measurement of ego pose relative to the lane.

    Missing lane geometry is structural: every metric field is ``None``
    instead of a fabricated zero-valued lane pose.
    """

    detected: bool
    lateral_error_m: float | None
    lateral_error_std_m: float | None
    heading_error_rad: float | None
    heading_error_std_rad: float | None
    curvature_inv_m: float | None
    curvature_std_inv_m: float | None
    visible_point_count: int
    fit_residual_m: float | None

    def __post_init__(self) -> None:
        values = (
            self.lateral_error_m,
            self.lateral_error_std_m,
            self.heading_error_rad,
            self.heading_error_std_rad,
            self.curvature_inv_m,
            self.curvature_std_inv_m,
            self.fit_residual_m,
        )
        if self.detected and any(value is None for value in values):
            raise ValueError("detected lane must have a complete measurement")
        if not self.detected and any(value is not None for value in values):
            raise ValueError("missing lane must use None for metric values")
        if self.visible_point_count < 0:
            raise ValueError("visible lane point count cannot be negative")
        present = tuple(value for value in values if value is not None)
        if not all(isfinite(value) for value in present):
            raise ValueError("lane measurement values must be finite")
        for value in (
            self.lateral_error_std_m,
            self.heading_error_std_rad,
            self.curvature_std_inv_m,
            self.fit_residual_m,
        ):
            if value is not None and value < 0.0:
                raise ValueError("lane uncertainty/residual cannot be negative")

    @classmethod
    def missing(cls, *, visible_point_count: int = 0) -> "LaneMeasurement":
        return cls(
            detected=False,
            lateral_error_m=None,
            lateral_error_std_m=None,
            heading_error_rad=None,
            heading_error_std_rad=None,
            curvature_inv_m=None,
            curvature_std_inv_m=None,
            visible_point_count=visible_point_count,
            fit_residual_m=None,
        )


@dataclass(frozen=True)
class ObjectMeasurement:
    """One-frame metric observation; ``range_m`` targets the object origin.

    A detector-facing raw range may require a fixed offline calibration before
    it is used here. Missing values are represented by ``None``.
    """

    object_class: ObjectClass
    detected: bool
    confidence: float | None
    x_left_m: float | None
    y_forward_m: float | None
    range_m: float | None
    bearing_rad: float | None

    def __post_init__(self) -> None:
        values = (
            self.confidence,
            self.x_left_m,
            self.y_forward_m,
            self.range_m,
            self.bearing_rad,
        )
        if self.detected and any(value is None for value in values):
            raise ValueError("detected object must have a complete measurement")
        if not self.detected and any(value is not None for value in values):
            raise ValueError("missed detection must use None for measurement values")
        present_values = [value for value in values if value is not None]
        if not all(isfinite(value) for value in present_values):
            raise ValueError("object measurement values must be finite")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("measurement confidence must be within [0, 1]")
        if self.range_m is not None and self.range_m < 0.0:
            raise ValueError("measurement range cannot be negative")

    @classmethod
    def missing(cls, object_class: ObjectClass) -> ObjectMeasurement:
        return cls(
            object_class=object_class,
            detected=False,
            confidence=None,
            x_left_m=None,
            y_forward_m=None,
            range_m=None,
            bearing_rad=None,
        )


@dataclass(frozen=True)
class PerceptionObservation:
    """Structured observation consumed by the belief updater."""

    ego: EgoObservation
    road: RoadMeasurement | None
    stop_sign: ObjectMeasurement
    pedestrian: ObjectMeasurement
    lane: LaneMeasurement | None = None

    def __post_init__(self) -> None:
        if self.stop_sign.object_class is not ObjectClass.STOP_SIGN:
            raise ValueError("stop_sign must contain a stop-sign measurement")
        if self.pedestrian.object_class is not ObjectClass.DUCKIE:
            raise ValueError("pedestrian must contain a Duckie measurement")
