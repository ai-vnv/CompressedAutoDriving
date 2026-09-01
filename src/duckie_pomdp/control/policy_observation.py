"""Semantic, leak-free policy input and fixed physical normalization."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.control.f10_protocol import F10Protocol
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.belief import BeliefState


@dataclass(frozen=True)
class PolicyObservation:
    lateral_error_m: float
    heading_error_rad: float
    actual_linear_velocity_mps: float
    actual_yaw_rate_rad_s: float
    road_curvature_inv_m: float
    stop_line_distance_m: float
    pedestrian_existence_probability: float
    pedestrian_range_mean_m: float
    pedestrian_range_std_m: float
    pedestrian_bearing_mean_rad: float
    pedestrian_bearing_std_rad: float
    pedestrian_radial_velocity_mean_mps: float
    pedestrian_radial_velocity_std_mps: float
    pedestrian_bearing_rate_mean_rad_s: float
    pedestrian_bearing_rate_std_rad_s: float
    previous_linear_velocity_cmd_mps: float
    previous_angular_velocity_cmd_rad_s: float

    def __post_init__(self) -> None:
        values = tuple(getattr(self, item.name) for item in fields(self))
        if not all(isfinite(value) for value in values):
            raise ValueError("policy observation values must be finite")
        if not 0.0 <= self.pedestrian_existence_probability <= 1.0:
            raise ValueError("pedestrian existence probability must be within [0, 1]")
        if self.pedestrian_range_mean_m < 0.0:
            raise ValueError("pedestrian range mean cannot be negative")
        standard_deviations = (
            self.pedestrian_range_std_m,
            self.pedestrian_bearing_std_rad,
            self.pedestrian_radial_velocity_std_mps,
            self.pedestrian_bearing_rate_std_rad_s,
        )
        if any(value < 0.0 for value in standard_deviations):
            raise ValueError("policy belief standard deviations cannot be negative")

    @classmethod
    def from_belief(
        cls,
        belief: BeliefState,
        previous_action: PolicyAction,
    ) -> PolicyObservation:
        pedestrian = belief.pedestrian
        return cls(
            lateral_error_m=belief.ego.lateral_error_m,
            heading_error_rad=belief.ego.heading_error_rad,
            actual_linear_velocity_mps=belief.ego.linear_velocity_mps,
            actual_yaw_rate_rad_s=belief.ego.yaw_rate_rad_s,
            road_curvature_inv_m=belief.road.curvature_inv_m,
            stop_line_distance_m=belief.road.stop_line_distance_m,
            pedestrian_existence_probability=pedestrian.existence_probability,
            pedestrian_range_mean_m=pedestrian.range_mean_m,
            pedestrian_range_std_m=pedestrian.range_std_m,
            pedestrian_bearing_mean_rad=pedestrian.bearing_mean_rad,
            pedestrian_bearing_std_rad=pedestrian.bearing_std_rad,
            pedestrian_radial_velocity_mean_mps=pedestrian.radial_velocity_mean_mps,
            pedestrian_radial_velocity_std_mps=pedestrian.radial_velocity_std_mps,
            pedestrian_bearing_rate_mean_rad_s=pedestrian.bearing_rate_mean_rad_s,
            pedestrian_bearing_rate_std_rad_s=pedestrian.bearing_rate_std_rad_s,
            previous_linear_velocity_cmd_mps=previous_action.linear_velocity_mps,
            previous_angular_velocity_cmd_rad_s=previous_action.angular_velocity_rad_s,
        )

    @classmethod
    def ordering(cls) -> tuple[str, ...]:
        return tuple(item.name for item in fields(cls))

    def to_vector(self) -> NDArray[np.float32]:
        return np.asarray(
            [getattr(self, name) for name in self.ordering()],
            dtype=np.float32,
        )


class FixedObservationNormalizer:
    """Stateless normalizer; evaluation has no statistics it can mutate."""

    def __init__(
        self,
        ordering: tuple[str, ...],
        scales: tuple[float, ...],
        clip: float,
    ) -> None:
        if ordering != PolicyObservation.ordering():
            raise ValueError("normalizer ordering differs from PolicyObservation")
        if len(scales) != len(ordering) or any(value <= 0.0 for value in scales):
            raise ValueError("normalizer needs one positive scale per feature")
        if not isfinite(clip) or clip <= 0.0:
            raise ValueError("normalization clip must be positive and finite")
        self._scales = np.asarray(scales, dtype=np.float32)
        self._clip = float(clip)

    @classmethod
    def from_protocol(cls, protocol: F10Protocol) -> FixedObservationNormalizer:
        return cls(
            protocol.observation_order,
            protocol.observation_scales,
            protocol.observation_clip,
        )

    @property
    def scales(self) -> NDArray[np.float32]:
        copy = self._scales.copy()
        copy.setflags(write=False)
        return copy

    def normalize(self, observation: PolicyObservation) -> NDArray[np.float32]:
        vector = observation.to_vector()
        normalized = np.clip(vector / self._scales, -self._clip, self._clip)
        return np.asarray(normalized, dtype=np.float32)

