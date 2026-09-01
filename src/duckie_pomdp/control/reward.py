"""Auditable F10 reward and episode semantics.

This is the only F10 runtime component allowed to consume simulator truth.
Its outputs are scalar training signals and evaluation diagnostics; the policy
observation is built independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, isfinite, sin
from typing import Any

from duckie_pomdp.control.f10_protocol import F10Protocol
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import SensorObservation
from duckie_pomdp.domain.privileged import PrivilegedSimulatorState, WorldFootprint


@dataclass(frozen=True)
class F10RewardConfig:
    progress_per_m: float
    living_penalty: float
    lane_lateral_weight: float
    lane_lateral_scale_m: float
    lane_heading_weight: float
    lane_heading_scale_rad: float
    lane_departure_threshold_m: float
    pedestrian_safety_distance_m: float
    pedestrian_unsafe_distance_m: float
    pedestrian_collision_distance_m: float
    pedestrian_proximity_weight: float
    pedestrian_unsafe_penalty: float
    comfort_linear_change_weight: float
    comfort_angular_change_weight: float
    comfort_steering_weight: float
    stop_approach_distance_m: float
    stop_speed_threshold_mps: float
    stop_hold_steps: int
    stop_approach_speed_weight: float
    stop_completion_bonus: float
    stop_violation_penalty: float
    pedestrian_collision_terminal_penalty: float
    invalid_pose_terminal_penalty: float
    success_terminal_bonus: float
    success_progress_m: float
    maximum_linear_velocity_mps: float
    maximum_angular_velocity_rad_s: float
    lane_departure_terminal_penalty: float | None = None
    pedestrian_stationary_proximity_factor: float = 0.25
    pedestrian_safe_clearance_bonus: float = 0.0

    @classmethod
    def from_protocol(cls, protocol: F10Protocol) -> F10RewardConfig:
        reward: dict[str, Any] = protocol.raw["reward"]
        _, maximum_linear, _, maximum_angular = protocol.action_bounds
        configured = {
            name: reward[name]
            for name in cls.__dataclass_fields__
            if name in reward
        }
        return cls(
            **configured,
            maximum_linear_velocity_mps=maximum_linear,
            maximum_angular_velocity_rad_s=maximum_angular,
        )


@dataclass(frozen=True)
class F10RewardTerms:
    progress: float
    lane: float
    stop: float
    pedestrian: float
    comfort: float
    terminal: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in self.as_tuple()):
            raise ValueError("reward terms must be finite")

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.progress,
            self.lane,
            self.stop,
            self.pedestrian,
            self.comfort,
            self.terminal,
        )

    @property
    def total(self) -> float:
        return sum(self.as_tuple())


@dataclass(frozen=True)
class F10StepOutcome:
    reward_terms: F10RewardTerms
    terminated: bool
    truncated: bool
    termination_reason: str | None
    truncation_reason: str | None
    progress_m: float
    pedestrian_clearance_m: float | None
    pedestrian_collision: bool
    unsafe_proximity: bool
    lane_departure: bool
    stop_completed: bool
    stop_violation: bool
    invalid_pose: bool

    @property
    def reward(self) -> float:
        return self.reward_terms.total


@dataclass(frozen=True)
class LoopHazardOutcome:
    """Object-only reward/events layered onto closed-loop lane navigation."""

    pedestrian_reward: float
    stop_reward: float
    terminal_reward: float
    terminated: bool
    termination_reason: str | None
    pedestrian_clearance_m: float | None
    pedestrian_collision: bool
    unsafe_proximity: bool
    stop_completed: bool
    stop_violation: bool


class LoopHazardRewardEvaluator:
    """Pedestrian/stop semantics without duplicating loop progress or lane reward.

    ``LaneRewardEvaluator`` remains the sole source of progress, lane,
    smoothness, and lap completion on ``experiment_loop``.  This class adds
    only the two object components and collision termination.
    """

    def __init__(
        self,
        config: F10RewardConfig,
        *,
        pedestrian_active: bool,
        stop_active: bool,
    ) -> None:
        self.config = config
        self.pedestrian_active = bool(pedestrian_active)
        self.stop_active = bool(stop_active)
        self._previous_stop_distance: float | None = None
        self._stop_hold_steps = 0
        self._stop_completed = False
        self._stop_violation = False
        self._pedestrian_encountered = False
        self._pedestrian_unsafe_seen = False
        self._pedestrian_clearance_awarded = False

    def reset(self, privileged: PrivilegedSimulatorState) -> None:
        self._previous_stop_distance = (
            _true_stop_distance(privileged) if self.stop_active else None
        )
        self._stop_hold_steps = 0
        self._stop_completed = False
        self._stop_violation = False
        clearance = (
            _ego_to_footprint_distance(privileged)
            if self.pedestrian_active
            else None
        )
        self._pedestrian_encountered = bool(
            clearance is not None
            and clearance < self.config.pedestrian_safety_distance_m
        )
        self._pedestrian_unsafe_seen = bool(
            clearance is not None
            and clearance <= self.config.pedestrian_unsafe_distance_m
        )
        self._pedestrian_clearance_awarded = False

    def evaluate(
        self,
        *,
        observation: SensorObservation,
        privileged: PrivilegedSimulatorState,
    ) -> LoopHazardOutcome:
        clearance = (
            _ego_to_footprint_distance(privileged)
            if self.pedestrian_active
            else None
        )
        collision = bool(
            clearance is not None
            and clearance <= self.config.pedestrian_collision_distance_m
        )
        unsafe = bool(
            clearance is not None
            and clearance <= self.config.pedestrian_unsafe_distance_m
        )
        if clearance is not None and clearance < self.config.pedestrian_safety_distance_m:
            self._pedestrian_encountered = True
        if unsafe:
            self._pedestrian_unsafe_seen = True
        pedestrian_reward = (
            self._pedestrian_reward(clearance, observation)
            if self.pedestrian_active
            else 0.0
        )
        if (
            clearance is not None
            and self._pedestrian_encountered
            and not self._pedestrian_unsafe_seen
            and not self._pedestrian_clearance_awarded
            and clearance >= self.config.pedestrian_safety_distance_m
        ):
            pedestrian_reward += self.config.pedestrian_safe_clearance_bonus
            self._pedestrian_clearance_awarded = True
        stop_reward = (
            self._stop_reward(observation, privileged)
            if self.stop_active
            else 0.0
        )
        if self.stop_active:
            self._previous_stop_distance = _true_stop_distance(privileged)
        return LoopHazardOutcome(
            pedestrian_reward=pedestrian_reward,
            stop_reward=stop_reward,
            terminal_reward=(
                self.config.pedestrian_collision_terminal_penalty
                if collision
                else 0.0
            ),
            terminated=collision,
            termination_reason="pedestrian_collision" if collision else None,
            pedestrian_clearance_m=clearance,
            pedestrian_collision=collision,
            unsafe_proximity=unsafe,
            stop_completed=self._stop_completed,
            stop_violation=self._stop_violation,
        )

    def _stop_reward(
        self,
        observation: SensorObservation,
        privileged: PrivilegedSimulatorState,
    ) -> float:
        distance = _true_stop_distance(privileged)
        in_zone = 0.0 <= distance <= self.config.stop_approach_distance_m
        reward = 0.0
        if in_zone and not self._stop_completed:
            normalized_speed = min(
                observation.ego.linear_velocity_mps
                / self.config.maximum_linear_velocity_mps,
                1.0,
            )
            reward += self.config.stop_approach_speed_weight * normalized_speed
            if (
                observation.ego.linear_velocity_mps
                <= self.config.stop_speed_threshold_mps
            ):
                self._stop_hold_steps += 1
                if self._stop_hold_steps >= self.config.stop_hold_steps:
                    self._stop_completed = True
                    reward += self.config.stop_completion_bonus
            else:
                self._stop_hold_steps = 0
        previous = self._previous_stop_distance
        if (
            previous is not None
            and previous > 0.0 >= distance
            and not self._stop_completed
            and not self._stop_violation
        ):
            self._stop_violation = True
            reward += self.config.stop_violation_penalty
        return reward

    def _pedestrian_reward(
        self,
        clearance: float | None,
        observation: SensorObservation,
    ) -> float:
        if (
            clearance is None
            or clearance >= self.config.pedestrian_safety_distance_m
        ):
            return 0.0
        penetration = 1.0 - clearance / self.config.pedestrian_safety_distance_m
        velocity_factor = self.config.pedestrian_stationary_proximity_factor + min(
            observation.ego.linear_velocity_mps
            / self.config.maximum_linear_velocity_mps,
            1.0,
        )
        reward = (
            self.config.pedestrian_proximity_weight
            * penetration
            * velocity_factor
        )
        if (
            clearance <= self.config.pedestrian_unsafe_distance_m
            and observation.ego.linear_velocity_mps
            > self.config.stop_speed_threshold_mps
        ):
            reward += self.config.pedestrian_unsafe_penalty
        return reward


class F10RewardEvaluator:
    """Stateful per-episode reward calculator with explicit event semantics."""

    def __init__(
        self,
        config: F10RewardConfig,
        *,
        route_heading_rad: float,
        pedestrian_active: bool = True,
        stop_active: bool = True,
    ) -> None:
        self.config = config
        self.route_heading_rad = float(route_heading_rad)
        self.pedestrian_active = bool(pedestrian_active)
        self.stop_active = bool(stop_active)
        self._initial_route_coordinate: float | None = None
        self._previous_route_coordinate: float | None = None
        self._previous_stop_distance: float | None = None
        self._previous_action = PolicyAction(0.0, 0.0)
        self._stop_hold_steps = 0
        self._stop_completed = False
        self._stop_violation = False

    def reset(
        self,
        observation: SensorObservation,
        privileged: PrivilegedSimulatorState,
    ) -> None:
        if observation.road is None:
            raise ValueError("F10 reward requires agent-visible road measurement")
        coordinate = self._route_coordinate(privileged)
        self._initial_route_coordinate = coordinate
        self._previous_route_coordinate = coordinate
        self._previous_stop_distance = (
            _true_stop_distance(privileged) if self.stop_active else None
        )
        self._previous_action = PolicyAction(0.0, 0.0)
        self._stop_hold_steps = 0
        self._stop_completed = False
        self._stop_violation = False

    def evaluate(
        self,
        *,
        action: PolicyAction,
        observation: SensorObservation,
        privileged: PrivilegedSimulatorState,
        simulator_terminated: bool,
        simulator_truncated: bool,
        simulator_done_code: str,
        horizon_reached: bool,
    ) -> F10StepOutcome:
        if self._initial_route_coordinate is None or self._previous_route_coordinate is None:
            raise RuntimeError("reward evaluator must be reset before evaluate")
        if observation.road is None:
            raise ValueError("F10 reward requires agent-visible road measurement")

        coordinate = self._route_coordinate(privileged)
        delta_progress = coordinate - self._previous_route_coordinate
        progress = coordinate - self._initial_route_coordinate
        terms_progress = self.config.progress_per_m * delta_progress + self.config.living_penalty

        lateral_ratio = observation.ego.lateral_error_m / self.config.lane_lateral_scale_m
        heading_ratio = observation.ego.heading_error_rad / self.config.lane_heading_scale_rad
        terms_lane = (
            self.config.lane_lateral_weight * lateral_ratio * lateral_ratio
            + self.config.lane_heading_weight * heading_ratio * heading_ratio
        )
        lane_departure = abs(observation.ego.lateral_error_m) > self.config.lane_departure_threshold_m

        terms_stop = (
            self._stop_reward(observation, privileged) if self.stop_active else 0.0
        )
        clearance = (
            _ego_to_footprint_distance(privileged) if self.pedestrian_active else None
        )
        collision = bool(
            clearance is not None
            and clearance <= self.config.pedestrian_collision_distance_m
        )
        unsafe = bool(
            clearance is not None
            and clearance <= self.config.pedestrian_unsafe_distance_m
        )
        terms_pedestrian = (
            self._pedestrian_reward(clearance, observation)
            if self.pedestrian_active
            else 0.0
        )

        linear_change = (
            action.linear_velocity_mps - self._previous_action.linear_velocity_mps
        ) / self.config.maximum_linear_velocity_mps
        angular_change = (
            action.angular_velocity_rad_s - self._previous_action.angular_velocity_rad_s
        ) / self.config.maximum_angular_velocity_rad_s
        normalized_steering = action.angular_velocity_rad_s / self.config.maximum_angular_velocity_rad_s
        terms_comfort = (
            self.config.comfort_linear_change_weight * linear_change * linear_change
            + self.config.comfort_angular_change_weight * angular_change * angular_change
            + self.config.comfort_steering_weight * normalized_steering * normalized_steering
        )

        invalid_pose = simulator_done_code == "invalid-pose"
        success = progress >= self.config.success_progress_m and (
            not self.stop_active or self._stop_completed
        )
        terminated = (
            collision
            or lane_departure
            or success
            or (simulator_terminated and not simulator_truncated)
        )
        truncated = bool(simulator_truncated or horizon_reached)
        termination_reason = None
        truncation_reason = None
        terminal_reward = 0.0
        if collision:
            termination_reason = "pedestrian_collision"
            terminal_reward = self.config.pedestrian_collision_terminal_penalty
        elif lane_departure:
            termination_reason = "lane_departure"
            terminal_reward = (
                self.config.invalid_pose_terminal_penalty
                if self.config.lane_departure_terminal_penalty is None
                else self.config.lane_departure_terminal_penalty
            )
        elif success:
            termination_reason = "success"
            terminal_reward = self.config.success_terminal_bonus
        elif simulator_terminated:
            termination_reason = "invalid_pose" if invalid_pose else f"simulator:{simulator_done_code}"
            if invalid_pose:
                terminal_reward = self.config.invalid_pose_terminal_penalty
        if truncated:
            truncation_reason = "horizon" if horizon_reached else f"simulator:{simulator_done_code}"
            if not (collision or lane_departure or success):
                terminated = False
                termination_reason = None

        self._previous_route_coordinate = coordinate
        self._previous_stop_distance = (
            _true_stop_distance(privileged) if self.stop_active else None
        )
        self._previous_action = action
        return F10StepOutcome(
            reward_terms=F10RewardTerms(
                progress=terms_progress,
                lane=terms_lane,
                stop=terms_stop,
                pedestrian=terms_pedestrian,
                comfort=terms_comfort,
                terminal=terminal_reward,
            ),
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
            truncation_reason=truncation_reason,
            progress_m=progress,
            pedestrian_clearance_m=clearance,
            pedestrian_collision=collision,
            unsafe_proximity=unsafe,
            lane_departure=lane_departure,
            stop_completed=self._stop_completed,
            stop_violation=self._stop_violation,
            invalid_pose=invalid_pose,
        )

    def _route_coordinate(self, privileged: PrivilegedSimulatorState) -> float:
        pose = privileged.ego_world_pose
        return pose.x_m * cos(self.route_heading_rad) - pose.z_m * sin(self.route_heading_rad)

    def _stop_reward(
        self,
        observation: SensorObservation,
        privileged: PrivilegedSimulatorState,
    ) -> float:
        distance = _true_stop_distance(privileged)
        in_zone = 0.0 <= distance <= self.config.stop_approach_distance_m
        reward = 0.0
        if in_zone and not self._stop_completed:
            normalized_speed = min(
                observation.ego.linear_velocity_mps / self.config.maximum_linear_velocity_mps,
                1.0,
            )
            reward += self.config.stop_approach_speed_weight * normalized_speed
            if observation.ego.linear_velocity_mps <= self.config.stop_speed_threshold_mps:
                self._stop_hold_steps += 1
                if self._stop_hold_steps >= self.config.stop_hold_steps:
                    self._stop_completed = True
                    reward += self.config.stop_completion_bonus
            else:
                self._stop_hold_steps = 0
        previous = self._previous_stop_distance
        if previous is not None and previous > 0.0 >= distance and not self._stop_completed and not self._stop_violation:
            self._stop_violation = True
            reward += self.config.stop_violation_penalty
        return reward

    def _pedestrian_reward(
        self,
        clearance: float | None,
        observation: SensorObservation,
    ) -> float:
        if clearance is None or clearance >= self.config.pedestrian_safety_distance_m:
            return 0.0
        penetration = 1.0 - clearance / self.config.pedestrian_safety_distance_m
        velocity_factor = self.config.pedestrian_stationary_proximity_factor + min(
            observation.ego.linear_velocity_mps / self.config.maximum_linear_velocity_mps,
            1.0,
        )
        reward = self.config.pedestrian_proximity_weight * penetration * velocity_factor
        if (
            clearance <= self.config.pedestrian_unsafe_distance_m
            and observation.ego.linear_velocity_mps > self.config.stop_speed_threshold_mps
        ):
            reward += self.config.pedestrian_unsafe_penalty
        return reward


def _ego_to_footprint_distance(privileged: PrivilegedSimulatorState) -> float | None:
    footprint = privileged.pedestrian_world_footprint
    if footprint is None:
        return None
    return _point_to_polygon_distance(
        privileged.ego_world_pose.x_m,
        privileged.ego_world_pose.z_m,
        footprint,
    )


def _true_stop_distance(privileged: PrivilegedSimulatorState) -> float:
    distance = privileged.true_pomdp_state.road.stop_line_distance_m
    if distance is None or not isfinite(distance):
        raise RuntimeError("privileged stop-line truth is unavailable to the reward")
    return distance


def _point_to_polygon_distance(x: float, z: float, footprint: WorldFootprint) -> float:
    vertices = footprint.vertices
    inside = False
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        if (first.z_m > z) != (second.z_m > z):
            x_cross = (second.x_m - first.x_m) * (z - first.z_m) / (second.z_m - first.z_m) + first.x_m
            if x < x_cross:
                inside = not inside
    if inside:
        return 0.0
    return min(
        _point_to_segment_distance(x, z, first.x_m, first.z_m, second.x_m, second.z_m)
        for first, second in zip(vertices, vertices[1:] + vertices[:1])
    )


def _point_to_segment_distance(
    x: float,
    z: float,
    x1: float,
    z1: float,
    x2: float,
    z2: float,
) -> float:
    dx, dz = x2 - x1, z2 - z1
    squared = dx * dx + dz * dz
    if squared == 0.0:
        return hypot(x - x1, z - z1)
    fraction = min(max(((x - x1) * dx + (z - z1) * dz) / squared, 0.0), 1.0)
    return hypot(x - (x1 + fraction * dx), z - (z1 + fraction * dz))
