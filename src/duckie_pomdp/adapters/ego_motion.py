"""Estimate actual chassis motion from Gym-Duckietown pose samples."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isfinite, sin

from duckie_pomdp.domain.state import EgoMotion


@dataclass(frozen=True)
class SimulatorPoseSample:
    """Pose in Gym-Duckietown's ``(x, z, heading)`` representation."""

    x_m: float
    z_m: float
    heading_rad: float
    timestamp_s: float

    def __post_init__(self) -> None:
        values = (self.x_m, self.z_m, self.heading_rad, self.timestamp_s)
        if not all(isfinite(value) for value in values):
            raise ValueError("pose sample values must be finite")


def estimate_actual_motion(
    previous: SimulatorPoseSample,
    current: SimulatorPoseSample,
) -> EgoMotion:
    """Estimate signed forward speed and CCW-positive yaw rate.

    Gym-Duckietown stores the ground plane as ``(x, z)`` while its SE(2)
    heading is Cartesian. A heading ``h`` therefore points along
    ``(cos(h), -sin(h))`` in the simulator's ``(x, z)`` coordinates.
    """

    dt_s = current.timestamp_s - previous.timestamp_s
    if dt_s <= 0.0:
        raise ValueError("pose timestamps must increase")

    delta_heading = _wrap_angle(current.heading_rad - previous.heading_rad)
    midpoint_heading = previous.heading_rad + 0.5 * delta_heading
    delta_x = current.x_m - previous.x_m
    delta_z = current.z_m - previous.z_m
    forward_distance = (
        delta_x * cos(midpoint_heading) - delta_z * sin(midpoint_heading)
    )
    return EgoMotion(
        linear_velocity_mps=forward_distance / dt_s,
        yaw_rate_rad_s=delta_heading / dt_s,
    )


def _wrap_angle(angle_rad: float) -> float:
    return atan2(sin(angle_rad), cos(angle_rad))
