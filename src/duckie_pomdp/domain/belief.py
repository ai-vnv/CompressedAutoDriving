"""Probabilistic belief exposed to the policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .observation import EgoObservation
from .state import StopMode


@dataclass(frozen=True)
class RoadBelief:
    """Effectively observed road quantities plus the stop-state machine."""

    curvature_inv_m: float
    stop_line_distance_m: float
    stop_mode: StopMode


@dataclass(frozen=True)
class LaneBelief:
    """Gaussian lane-pose belief produced from front-camera road markings."""

    validity_probability: float
    lateral_error_mean_m: float
    lateral_error_std_m: float
    heading_error_mean_rad: float
    heading_error_std_rad: float
    curvature_mean_inv_m: float
    curvature_std_inv_m: float

    def __post_init__(self) -> None:
        values = (
            self.validity_probability,
            self.lateral_error_mean_m,
            self.lateral_error_std_m,
            self.heading_error_mean_rad,
            self.heading_error_std_rad,
            self.curvature_mean_inv_m,
            self.curvature_std_inv_m,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("lane belief values must be finite")
        if not 0.0 <= self.validity_probability <= 1.0:
            raise ValueError("lane validity probability must be within [0, 1]")
        if any(
            value < 0.0
            for value in (
                self.lateral_error_std_m,
                self.heading_error_std_rad,
                self.curvature_std_inv_m,
            )
        ):
            raise ValueError("lane belief standard deviations cannot be negative")


@dataclass(frozen=True)
class StopSignBelief:
    existence_probability: float
    range_mean_m: float
    range_std_m: float
    bearing_mean_rad: float
    bearing_std_rad: float

    def __post_init__(self) -> None:
        _validate_belief(
            self.existence_probability,
            means=(self.range_mean_m, self.bearing_mean_rad),
            standard_deviations=(self.range_std_m, self.bearing_std_rad),
        )


@dataclass(frozen=True)
class PedestrianBelief:
    """Public polar belief; both rates are relative to the moving ego."""

    existence_probability: float
    range_mean_m: float
    range_std_m: float
    bearing_mean_rad: float
    bearing_std_rad: float
    radial_velocity_mean_mps: float
    radial_velocity_std_mps: float
    bearing_rate_mean_rad_s: float
    bearing_rate_std_rad_s: float

    def __post_init__(self) -> None:
        _validate_belief(
            self.existence_probability,
            means=(
                self.range_mean_m,
                self.bearing_mean_rad,
                self.radial_velocity_mean_mps,
                self.bearing_rate_mean_rad_s,
            ),
            standard_deviations=(
                self.range_std_m,
                self.bearing_std_rad,
                self.radial_velocity_std_mps,
                self.bearing_rate_std_rad_s,
            ),
        )


@dataclass(frozen=True)
class BeliefState:
    """Fixed semantic policy state before numerical vector encoding."""

    ego: EgoObservation
    road: RoadBelief
    stop_sign: StopSignBelief
    pedestrian: PedestrianBelief
    # Optional only for frozen F0--F10-v1 artifact compatibility.  The visual
    # PPO v2 boundary rejects a missing lane belief.
    lane: LaneBelief | None = None


def _validate_belief(
    existence_probability: float,
    *,
    means: tuple[float, ...],
    standard_deviations: tuple[float, ...],
) -> None:
    if not isfinite(existence_probability):
        raise ValueError("existence probability must be finite")
    if not 0.0 <= existence_probability <= 1.0:
        raise ValueError("existence probability must be within [0, 1]")
    if not all(isfinite(value) for value in means + standard_deviations):
        raise ValueError("belief moments must be finite")
    if any(value < 0.0 for value in standard_deviations):
        raise ValueError("belief standard deviations cannot be negative")
    if means and means[0] < 0.0:
        raise ValueError("mean range cannot be negative")
