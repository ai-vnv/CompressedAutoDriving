"""SE(2) compensation for a Cartesian state stored in the moving ego frame."""

from __future__ import annotations

from math import cos, isfinite, sin

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.domain.state import EgoMotion


def ego_axis_rotation(delta_yaw_rad: float) -> NDArray[np.float64]:
    """Rotate old ego-axis components into the new ego-oriented axes."""

    if not isfinite(delta_yaw_rad):
        raise ValueError("ego yaw increment must be finite")
    cosine = cos(delta_yaw_rad)
    sine = sin(delta_yaw_rad)
    # Coordinates are [left, forward]. Positive ego yaw makes an object that
    # was straight ahead move toward negative x (right) in the new frame.
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def ego_displacement_old_axes(
    ego_motion: EgoMotion,
    dt_s: float,
) -> NDArray[np.float64]:
    """Constant-twist ego displacement as [left, forward] in old axes."""

    _validate_motion(ego_motion, dt_s)
    speed = ego_motion.linear_velocity_mps
    yaw_rate = ego_motion.yaw_rate_rad_s
    delta_yaw = yaw_rate * dt_s
    if abs(yaw_rate) <= 1.0e-9:
        return np.array([0.0, speed * dt_s], dtype=float)
    radius = speed / yaw_rate
    return np.array(
        [radius * (1.0 - cos(delta_yaw)), radius * sin(delta_yaw)],
        dtype=float,
    )


def compensated_transition(
    ego_motion: EgoMotion,
    dt_s: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return affine ``state_new = F @ state_old + offset``.

    State velocity is pedestrian physical velocity expressed in ego-oriented
    axes. It is rotated with the frame but never subtracts ego velocity.
    """

    _validate_motion(ego_motion, dt_s)
    rotation = ego_axis_rotation(ego_motion.yaw_rate_rad_s * dt_s)
    displacement = ego_displacement_old_axes(ego_motion, dt_s)
    transition = np.zeros((4, 4), dtype=float)
    transition[:2, :2] = rotation
    transition[:2, 2:] = dt_s * rotation
    transition[2:, 2:] = rotation
    offset = np.zeros(4, dtype=float)
    offset[:2] = -rotation @ displacement
    return transition, offset


def _validate_motion(ego_motion: EgoMotion, dt_s: float) -> None:
    values = (
        ego_motion.linear_velocity_mps,
        ego_motion.yaw_rate_rad_s,
        dt_s,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("ego motion and dt must be finite")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

