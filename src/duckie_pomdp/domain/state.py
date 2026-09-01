"""Conceptual POMDP state and actual ego-motion contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class StopMode(str, Enum):
    """Stop-compliance state for the single Version-1 stopping location."""

    NONE = "none"
    REQUIRED = "required"
    SATISFIED = "satisfied"


@dataclass(frozen=True)
class EgoState:
    """Actual ego state; its velocity fields are not policy commands."""

    lateral_error_m: float
    heading_error_rad: float
    linear_velocity_mps: float
    yaw_rate_rad_s: float


@dataclass(frozen=True)
class EgoMotion:
    """Actual chassis motion used for ego-motion compensation."""

    linear_velocity_mps: float
    yaw_rate_rad_s: float


@dataclass(frozen=True)
class RoadState:
    """True lane geometry and stop obligation along the selected route."""

    curvature_inv_m: float | None
    stop_line_distance_m: float | None
    stop_mode: StopMode


@dataclass(frozen=True)
class StopSignState:
    """True ego-relative state of the stop-sign model origin."""

    exists: bool
    range_m: float | None
    bearing_rad: float | None

    def __post_init__(self) -> None:
        _validate_optional_object_state(
            exists=self.exists,
            values=(self.range_m, self.bearing_rad),
            object_name="stop sign",
        )


@dataclass(frozen=True)
class PedestrianState:
    """True ego-relative state of the Duckie model origin."""

    exists: bool
    range_m: float | None
    bearing_rad: float | None
    radial_velocity_mps: float | None
    bearing_rate_rad_s: float | None

    def __post_init__(self) -> None:
        _validate_optional_object_state(
            exists=self.exists,
            values=(
                self.range_m,
                self.bearing_rad,
                self.radial_velocity_mps,
                self.bearing_rate_rad_s,
            ),
            object_name="pedestrian",
        )


@dataclass(frozen=True)
class POMDPState:
    """Conceptual true state used by the simulator and oracle evaluation."""

    ego: EgoState
    road: RoadState
    stop_sign: StopSignState
    pedestrian: PedestrianState


def _validate_optional_object_state(
    *,
    exists: bool,
    values: tuple[float | None, ...],
    object_name: str,
) -> None:
    if exists and any(value is None for value in values):
        raise ValueError(f"existing {object_name} must have a complete state")
    if not exists and any(value is not None for value in values):
        raise ValueError(f"absent {object_name} cannot have kinematic values")
    present_values = [value for value in values if value is not None]
    if not all(isfinite(value) for value in present_values):
        raise ValueError(f"{object_name} state must be finite")
    if exists and values[0] is not None and values[0] < 0.0:
        raise ValueError(f"{object_name} range cannot be negative")
