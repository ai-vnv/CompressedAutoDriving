"""Fixed semantic policy representation shared by every PPO stage."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.belief import BeliefState, PedestrianBelief, StopSignBelief
from duckie_pomdp.domain.state import StopMode

from .ppo_protocol import PPOCurriculumProtocol


@dataclass(frozen=True)
class PPOPolicyObservation:
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
    stop_sign_existence_probability: float
    stop_sign_range_mean_m: float
    stop_sign_range_std_m: float
    stop_sign_bearing_mean_rad: float
    stop_sign_bearing_std_rad: float
    stop_mode_none: float
    stop_mode_required: float
    stop_mode_satisfied: float
    previous_linear_velocity_cmd_mps: float
    previous_angular_velocity_cmd_rad_s: float

    def __post_init__(self) -> None:
        values = self.to_vector()
        if not np.all(np.isfinite(values)):
            raise ValueError("PPO policy observation must be finite")
        for value in (
            self.pedestrian_existence_probability,
            self.stop_sign_existence_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("existence probability must be within [0,1]")
        one_hot = (self.stop_mode_none, self.stop_mode_required, self.stop_mode_satisfied)
        if one_hot not in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            raise ValueError("stop mode must be a valid one-hot value")

    @classmethod
    def from_belief(
        cls,
        belief: BeliefState,
        previous_action: PolicyAction,
    ) -> PPOPolicyObservation:
        pedestrian = belief.pedestrian
        sign = belief.stop_sign
        mode = belief.road.stop_mode
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
            stop_sign_existence_probability=sign.existence_probability,
            stop_sign_range_mean_m=sign.range_mean_m,
            stop_sign_range_std_m=sign.range_std_m,
            stop_sign_bearing_mean_rad=sign.bearing_mean_rad,
            stop_sign_bearing_std_rad=sign.bearing_std_rad,
            stop_mode_none=float(mode is StopMode.NONE),
            stop_mode_required=float(mode is StopMode.REQUIRED),
            stop_mode_satisfied=float(mode is StopMode.SATISFIED),
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


@dataclass(frozen=True)
class PPOVisualPolicyObservation:
    """Visual-lane belief boundary used by the retrained PPO v2 curriculum."""

    lane_validity_probability: float
    lane_lateral_error_mean_m: float
    lane_lateral_error_std_m: float
    lane_heading_error_mean_rad: float
    lane_heading_error_std_rad: float
    actual_linear_velocity_mps: float
    actual_yaw_rate_rad_s: float
    lane_curvature_mean_inv_m: float
    lane_curvature_std_inv_m: float
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
    stop_sign_existence_probability: float
    stop_sign_range_mean_m: float
    stop_sign_range_std_m: float
    stop_sign_bearing_mean_rad: float
    stop_sign_bearing_std_rad: float
    stop_mode_none: float
    stop_mode_required: float
    stop_mode_satisfied: float
    previous_linear_velocity_cmd_mps: float
    previous_angular_velocity_cmd_rad_s: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(self.to_vector())):
            raise ValueError("visual PPO policy observation must be finite")
        for value in (
            self.lane_validity_probability,
            self.pedestrian_existence_probability,
            self.stop_sign_existence_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("belief probability must be within [0,1]")
        one_hot = (self.stop_mode_none, self.stop_mode_required, self.stop_mode_satisfied)
        if one_hot not in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            raise ValueError("stop mode must be a valid one-hot value")

    @classmethod
    def from_belief(
        cls,
        belief: BeliefState,
        previous_action: PolicyAction,
    ) -> PPOVisualPolicyObservation:
        lane = belief.lane
        if lane is None:
            raise ValueError("visual PPO v2 requires an explicit LaneBelief")
        pedestrian = belief.pedestrian
        sign = belief.stop_sign
        mode = belief.road.stop_mode
        return cls(
            lane_validity_probability=lane.validity_probability,
            lane_lateral_error_mean_m=lane.lateral_error_mean_m,
            lane_lateral_error_std_m=lane.lateral_error_std_m,
            lane_heading_error_mean_rad=lane.heading_error_mean_rad,
            lane_heading_error_std_rad=lane.heading_error_std_rad,
            actual_linear_velocity_mps=belief.ego.linear_velocity_mps,
            actual_yaw_rate_rad_s=belief.ego.yaw_rate_rad_s,
            lane_curvature_mean_inv_m=lane.curvature_mean_inv_m,
            lane_curvature_std_inv_m=lane.curvature_std_inv_m,
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
            stop_sign_existence_probability=sign.existence_probability,
            stop_sign_range_mean_m=sign.range_mean_m,
            stop_sign_range_std_m=sign.range_std_m,
            stop_sign_bearing_mean_rad=sign.bearing_mean_rad,
            stop_sign_bearing_std_rad=sign.bearing_std_rad,
            stop_mode_none=float(mode is StopMode.NONE),
            stop_mode_required=float(mode is StopMode.REQUIRED),
            stop_mode_satisfied=float(mode is StopMode.SATISFIED),
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


class PPOFixedObservationNormalizer:
    """Immutable physical normalization; evaluation cannot update it."""

    def __init__(self, protocol: PPOCurriculumProtocol) -> None:
        self._observation_type = policy_observation_type(protocol)
        if protocol.observation_order != self._observation_type.ordering():
            raise ValueError("config ordering differs from policy observation type")
        if not isfinite(protocol.observation_clip) or protocol.observation_clip <= 0.0:
            raise ValueError("normalization clip must be finite and positive")
        self._scales = np.asarray(protocol.observation_scales, dtype=np.float32)
        self._clip = float(protocol.observation_clip)

    @property
    def scales(self) -> NDArray[np.float32]:
        result = self._scales.copy()
        result.setflags(write=False)
        return result

    def normalize(
        self, observation: PPOPolicyObservation | PPOVisualPolicyObservation
    ) -> NDArray[np.float32]:
        if not isinstance(observation, self._observation_type):
            raise TypeError("policy observation representation/config mismatch")
        return np.asarray(
            np.clip(observation.to_vector() / self._scales, -self._clip, self._clip),
            dtype=np.float32,
        )


def policy_observation_type(
    protocol: PPOCurriculumProtocol,
) -> type[PPOPolicyObservation] | type[PPOVisualPolicyObservation]:
    representation = str(
        protocol.raw["observation"].get("representation", "legacy_state_v1")
    )
    if representation == "legacy_state_v1":
        return PPOPolicyObservation
    if representation == "visual_lane_belief_v2":
        return PPOVisualPolicyObservation
    raise ValueError(f"unsupported PPO observation representation: {representation}")


def policy_observation_from_belief(
    protocol: PPOCurriculumProtocol,
    belief: BeliefState,
    previous_action: PolicyAction,
) -> PPOPolicyObservation | PPOVisualPolicyObservation:
    observation = policy_observation_type(protocol).from_belief(
        belief, previous_action
    )
    threshold = protocol.raw["observation"].get(
        "pedestrian_kinematics_min_existence_probability"
    )
    if threshold is None or not isinstance(observation, PPOVisualPolicyObservation):
        return observation
    minimum = float(threshold)
    if not 0.0 <= minimum <= 1.0:
        raise ValueError(
            "pedestrian kinematics existence threshold must be within [0,1]"
        )
    if observation.pedestrian_existence_probability >= minimum:
        return observation

    # The Gaussian kinematics are conditional on the object existing.  Once
    # the public existence belief is below the frozen validity threshold, stale
    # EKF means must not masquerade as a current pedestrian location.  Preserve
    # the fixed 29-D interface while encoding a semantically absent pedestrian.
    neutral = protocol.raw["neutral"]
    return replace(
        observation,
        pedestrian_existence_probability=float(
            neutral["pedestrian_existence_probability"]
        ),
        pedestrian_range_mean_m=float(neutral["pedestrian_range_mean_m"]),
        pedestrian_range_std_m=float(neutral["pedestrian_range_std_m"]),
        pedestrian_bearing_mean_rad=float(neutral["pedestrian_bearing_mean_rad"]),
        pedestrian_bearing_std_rad=float(neutral["pedestrian_bearing_std_rad"]),
        pedestrian_radial_velocity_mean_mps=float(
            neutral["pedestrian_radial_velocity_mean_mps"]
        ),
        pedestrian_radial_velocity_std_mps=float(
            neutral["pedestrian_radial_velocity_std_mps"]
        ),
        pedestrian_bearing_rate_mean_rad_s=float(
            neutral["pedestrian_bearing_rate_mean_rad_s"]
        ),
        pedestrian_bearing_rate_std_rad_s=float(
            neutral["pedestrian_bearing_rate_std_rad_s"]
        ),
    )


def neutral_pedestrian(protocol: PPOCurriculumProtocol) -> PedestrianBelief:
    values = protocol.raw["neutral"]
    return PedestrianBelief(
        existence_probability=float(values["pedestrian_existence_probability"]),
        range_mean_m=float(values["pedestrian_range_mean_m"]),
        range_std_m=float(values["pedestrian_range_std_m"]),
        bearing_mean_rad=float(values["pedestrian_bearing_mean_rad"]),
        bearing_std_rad=float(values["pedestrian_bearing_std_rad"]),
        radial_velocity_mean_mps=float(values["pedestrian_radial_velocity_mean_mps"]),
        radial_velocity_std_mps=float(values["pedestrian_radial_velocity_std_mps"]),
        bearing_rate_mean_rad_s=float(values["pedestrian_bearing_rate_mean_rad_s"]),
        bearing_rate_std_rad_s=float(values["pedestrian_bearing_rate_std_rad_s"]),
    )


def neutral_stop_sign(protocol: PPOCurriculumProtocol) -> StopSignBelief:
    values = protocol.raw["neutral"]
    return StopSignBelief(
        existence_probability=float(values["stop_sign_existence_probability"]),
        range_mean_m=float(values["stop_sign_range_mean_m"]),
        range_std_m=float(values["stop_sign_range_std_m"]),
        bearing_mean_rad=float(values["stop_sign_bearing_mean_rad"]),
        bearing_std_rad=float(values["stop_sign_bearing_std_rad"]),
    )
