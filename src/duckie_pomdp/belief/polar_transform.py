"""Cartesian posterior to public polar pedestrian-belief moments."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, isfinite, sqrt

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.domain.state import EgoMotion


@dataclass(frozen=True)
class PolarBeliefMoments:
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]

    @property
    def standard_deviation(self) -> NDArray[np.float64]:
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))


def cartesian_to_polar_state(
    state: NDArray[np.float64],
    ego_motion: EgoMotion,
    *,
    minimum_range_m: float = 1.0e-3,
) -> NDArray[np.float64]:
    """Return ``[r, beta, rdot, betadot]``.

    Cartesian velocity is pedestrian physical velocity. Relative range and
    bearing rates include actual ego translation/yaw analytically.
    """

    vector = _state_vector(state)
    x_left, y_forward, velocity_left, velocity_forward = vector
    range_m = hypot(x_left, y_forward)
    guarded_range = max(range_m, minimum_range_m)
    range_squared = guarded_range**2
    relative_forward_velocity = (
        velocity_forward - ego_motion.linear_velocity_mps
    )
    radial_velocity = (
        x_left * velocity_left + y_forward * relative_forward_velocity
    ) / guarded_range
    bearing_rate = (
        y_forward * velocity_left
        - x_left * relative_forward_velocity
    ) / range_squared - ego_motion.yaw_rate_rad_s
    return np.array(
        [
            range_m,
            atan2(x_left, y_forward),
            radial_velocity,
            bearing_rate,
        ],
        dtype=float,
    )


def polar_state_jacobian(
    state: NDArray[np.float64],
    ego_motion: EgoMotion,
    *,
    minimum_range_m: float = 1.0e-3,
) -> NDArray[np.float64]:
    """Analytical Jacobian of :func:`cartesian_to_polar_state`."""

    x_left, y_forward, velocity_left, velocity_forward = _state_vector(state)
    range_squared = x_left**2 + y_forward**2
    if range_squared < minimum_range_m**2:
        range_squared = minimum_range_m**2
    range_m = sqrt(range_squared)
    relative_forward_velocity = (
        velocity_forward - ego_motion.linear_velocity_mps
    )
    radial_numerator = (
        x_left * velocity_left + y_forward * relative_forward_velocity
    )
    bearing_numerator = (
        y_forward * velocity_left - x_left * relative_forward_velocity
    )

    jacobian = np.zeros((4, 4), dtype=float)
    jacobian[0, :2] = (x_left / range_m, y_forward / range_m)
    jacobian[1, :2] = (
        y_forward / range_squared,
        -x_left / range_squared,
    )
    jacobian[2] = (
        velocity_left / range_m
        - radial_numerator * x_left / range_m**3,
        relative_forward_velocity / range_m
        - radial_numerator * y_forward / range_m**3,
        x_left / range_m,
        y_forward / range_m,
    )
    jacobian[3] = (
        (
            -relative_forward_velocity * range_squared
            - 2.0 * x_left * bearing_numerator
        )
        / range_squared**2,
        (
            velocity_left * range_squared
            - 2.0 * y_forward * bearing_numerator
        )
        / range_squared**2,
        y_forward / range_squared,
        -x_left / range_squared,
    )
    return jacobian


def cartesian_to_polar_moments(
    state: NDArray[np.float64],
    covariance: NDArray[np.float64],
    ego_motion: EgoMotion,
    *,
    minimum_range_m: float = 1.0e-3,
) -> PolarBeliefMoments:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Cartesian covariance must be a finite 4x4 matrix")
    jacobian = polar_state_jacobian(
        state,
        ego_motion,
        minimum_range_m=minimum_range_m,
    )
    polar_covariance = jacobian @ matrix @ jacobian.T
    polar_covariance = 0.5 * (polar_covariance + polar_covariance.T)
    return PolarBeliefMoments(
        mean=cartesian_to_polar_state(
            state,
            ego_motion,
            minimum_range_m=minimum_range_m,
        ),
        covariance=polar_covariance,
    )


def _state_vector(state: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = np.asarray(state, dtype=float)
    if vector.shape != (4,) or not np.all(np.isfinite(vector)):
        raise ValueError("Cartesian pedestrian state must be a finite length-4 vector")
    return vector

