"""Six-dimensional agent-visible observation for F10-L1."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.control.lane_protocol import LaneProtocol
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import SensorObservation


@dataclass(frozen=True)
class LanePolicyObservation:
    lateral_error_m: float
    heading_error_rad: float
    actual_linear_velocity_mps: float
    actual_yaw_rate_rad_s: float
    previous_linear_velocity_cmd_mps: float
    previous_angular_velocity_cmd_rad_s: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in self.to_vector()):
            raise ValueError("lane-policy observation values must be finite")

    @classmethod
    def from_sensor(
        cls,
        observation: SensorObservation,
        previous_action: PolicyAction,
    ) -> LanePolicyObservation:
        ego = observation.ego
        return cls(
            lateral_error_m=ego.lateral_error_m,
            heading_error_rad=ego.heading_error_rad,
            actual_linear_velocity_mps=ego.linear_velocity_mps,
            actual_yaw_rate_rad_s=ego.yaw_rate_rad_s,
            previous_linear_velocity_cmd_mps=previous_action.linear_velocity_mps,
            previous_angular_velocity_cmd_rad_s=previous_action.angular_velocity_rad_s,
        )

    @classmethod
    def ordering(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    def to_vector(self) -> NDArray[np.float32]:
        return np.asarray(
            [getattr(self, name) for name in self.ordering()], dtype=np.float32
        )


class LaneObservationNormalizer:
    """Immutable, fixed-physical-scale normalization for F10-L1."""

    def __init__(
        self,
        ordering: tuple[str, ...],
        scales: tuple[float, ...],
        clip: float,
    ) -> None:
        if ordering != LanePolicyObservation.ordering():
            raise ValueError("normalizer ordering differs from lane observation")
        if len(scales) != len(ordering) or any(value <= 0.0 for value in scales):
            raise ValueError("normalizer requires one positive scale per feature")
        if not isfinite(clip) or clip <= 0.0:
            raise ValueError("normalization clip must be positive and finite")
        self._scales = np.asarray(scales, dtype=np.float32)
        self._clip = float(clip)

    @classmethod
    def from_protocol(cls, protocol: LaneProtocol) -> LaneObservationNormalizer:
        return cls(
            protocol.observation_order,
            protocol.observation_scales,
            protocol.observation_clip,
        )

    @property
    def scales(self) -> NDArray[np.float32]:
        result = self._scales.copy()
        result.setflags(write=False)
        return result

    def normalize(
        self, observation: LanePolicyObservation
    ) -> NDArray[np.float32]:
        vector = observation.to_vector()
        return np.asarray(
            np.clip(vector / self._scales, -self._clip, self._clip),
            dtype=np.float32,
        )

