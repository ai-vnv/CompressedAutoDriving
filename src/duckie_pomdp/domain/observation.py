"""Raw sensor and effectively observed ego/road quantities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .state import EgoMotion


@dataclass(frozen=True)
class EgoObservation:
    """Measured ego quantities available to the Version-1 agent."""

    lateral_error_m: float
    heading_error_rad: float
    linear_velocity_mps: float
    yaw_rate_rad_s: float

    @property
    def motion(self) -> EgoMotion:
        return EgoMotion(
            linear_velocity_mps=self.linear_velocity_mps,
            yaw_rate_rad_s=self.yaw_rate_rad_s,
        )


@dataclass(frozen=True)
class RoadMeasurement:
    """Measured route geometry; stop-sign range is intentionally absent."""

    curvature_inv_m: float
    stop_line_distance_m: float


@dataclass(frozen=True)
class SensorObservation:
    """Agent-visible camera and ego/road measurements for one timestep."""

    front_rgb: NDArray[np.uint8]
    ego: EgoObservation
    road: RoadMeasurement | None = None

    def __post_init__(self) -> None:
        if self.front_rgb.dtype != np.uint8:
            raise ValueError("front_rgb must use uint8 pixels")
        if self.front_rgb.ndim != 3 or self.front_rgb.shape[2] != 3:
            raise ValueError("front_rgb must have shape (height, width, 3)")
