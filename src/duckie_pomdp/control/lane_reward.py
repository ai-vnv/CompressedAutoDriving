"""Closed-loop lane reward and episode semantics for F10-L1."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, sin
from typing import Any

from duckie_pomdp.control.lane_protocol import LaneProtocol
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import SensorObservation
from duckie_pomdp.domain.privileged import WorldPose


@dataclass(frozen=True)
class LaneRewardConfig:
    target_velocity_mps: float
    progress_per_m: float
    living_penalty: float
    lane_lateral_weight: float
    lane_lateral_scale_m: float
    lane_heading_weight: float
    lane_heading_scale_rad: float
    yellow_warning_weight: float
    comfort_linear_change_weight: float
    comfort_angular_change_weight: float
    comfort_steering_weight: float
    lap_completion_bonus: float
    yellow_crossing_penalty: float
    lane_departure_penalty: float
    invalid_pose_penalty: float
    lane_center_to_yellow_m: float
    ego_half_width_m: float
    yellow_warning_clearance_m: float
    lane_departure_threshold_m: float
    minimum_path_length_m: float
    leave_start_radius_m: float
    finish_radius_m: float
    finish_heading_tolerance_rad: float
    maximum_linear_velocity_mps: float
    maximum_angular_velocity_rad_s: float
    yellow_curve_recovery_enabled: bool = False
    yellow_curve_min_abs_curvature_inv_m: float = 0.75
    yellow_curve_max_penetration_m: float = 0.035
    yellow_curve_exit_grace_steps: int = 15
    yellow_recovery_clear_steps: int = 3

    def __post_init__(self) -> None:
        if self.yellow_curve_min_abs_curvature_inv_m < 0.0:
            raise ValueError("yellow curve-curvature threshold cannot be negative")
        if self.yellow_curve_max_penetration_m < 0.0:
            raise ValueError("yellow curve penetration cannot be negative")
        if self.yellow_curve_exit_grace_steps < 0:
            raise ValueError("yellow curve recovery grace cannot be negative")
        if self.yellow_recovery_clear_steps <= 0:
            raise ValueError("yellow recovery clear steps must be positive")

    @classmethod
    def from_protocol(cls, protocol: LaneProtocol) -> LaneRewardConfig:
        reward = protocol.raw["reward"]
        geometry = protocol.raw["lane_geometry"]
        lap = protocol.raw["lap"]
        _, maximum_linear, _, maximum_angular = protocol.action_bounds
        return cls(
            **{name: float(reward[name]) for name in (
                "target_velocity_mps",
                "progress_per_m",
                "living_penalty",
                "lane_lateral_weight",
                "lane_lateral_scale_m",
                "lane_heading_weight",
                "lane_heading_scale_rad",
                "yellow_warning_weight",
                "comfort_linear_change_weight",
                "comfort_angular_change_weight",
                "comfort_steering_weight",
                "lap_completion_bonus",
                "yellow_crossing_penalty",
                "lane_departure_penalty",
                "invalid_pose_penalty",
            )},
            **{name: float(geometry[name]) for name in (
                "lane_center_to_yellow_m",
                "ego_half_width_m",
                "yellow_warning_clearance_m",
                "lane_departure_threshold_m",
            )},
            **{name: float(lap[name]) for name in (
                "minimum_path_length_m",
                "leave_start_radius_m",
                "finish_radius_m",
                "finish_heading_tolerance_rad",
            )},
            maximum_linear_velocity_mps=maximum_linear,
            maximum_angular_velocity_rad_s=maximum_angular,
        )


@dataclass(frozen=True)
class LaneRewardTerms:
    progress: float
    lane: float
    yellow: float
    comfort: float
    living: float
    terminal: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in self.as_tuple()):
            raise ValueError("lane reward terms must be finite")

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.progress,
            self.lane,
            self.yellow,
            self.comfort,
            self.living,
            self.terminal,
        )

    @property
    def total(self) -> float:
        return sum(self.as_tuple())


@dataclass(frozen=True)
class LaneStepOutcome:
    reward_terms: LaneRewardTerms
    terminated: bool
    truncated: bool
    termination_reason: str | None
    truncation_reason: str | None
    path_length_m: float
    yellow_clearance_m: float
    lap_completed: bool
    yellow_contact: bool
    yellow_crossing: bool
    yellow_recovery_started: bool
    yellow_recovery_active: bool
    yellow_recovered: bool
    lane_departure: bool
    invalid_pose: bool

    @property
    def reward(self) -> float:
        return self.reward_terms.total


class LaneRewardEvaluator:
    """Uses world pose only inside the reward/evaluation boundary."""

    def __init__(self, config: LaneRewardConfig, *, dt_s: float) -> None:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self.config = config
        self.dt_s = float(dt_s)
        self._start_pose: WorldPose | None = None
        self._previous_pose: WorldPose | None = None
        self._path_length_m = 0.0
        self._left_start = False
        self._previous_action = PolicyAction(0.0, 0.0)
        self._yellow_recovery_active = False
        self._yellow_recovery_grace_remaining = 0
        self._yellow_recovery_clear_streak = 0

    def reset(self, world_pose: WorldPose) -> None:
        self._start_pose = world_pose
        self._previous_pose = world_pose
        self._path_length_m = 0.0
        self._left_start = False
        self._previous_action = PolicyAction(0.0, 0.0)
        self._yellow_recovery_active = False
        self._yellow_recovery_grace_remaining = 0
        self._yellow_recovery_clear_streak = 0

    def evaluate(
        self,
        *,
        action: PolicyAction,
        observation: SensorObservation,
        world_pose: WorldPose,
        simulator_terminated: bool,
        simulator_truncated: bool,
        simulator_done_code: str,
        horizon_reached: bool,
        road_curvature_inv_m: float | None = None,
    ) -> LaneStepOutcome:
        if self._start_pose is None or self._previous_pose is None:
            raise RuntimeError("lane reward must be reset before evaluation")
        step_distance = hypot(
            world_pose.x_m - self._previous_pose.x_m,
            world_pose.z_m - self._previous_pose.z_m,
        )
        self._path_length_m += step_distance
        start_distance = hypot(
            world_pose.x_m - self._start_pose.x_m,
            world_pose.z_m - self._start_pose.z_m,
        )
        if start_distance > self.config.leave_start_radius_m:
            self._left_start = True

        ego = observation.ego
        alignment = max(cos(ego.heading_error_rad), 0.0)
        center_ratio = abs(ego.lateral_error_m) / self.config.lane_departure_threshold_m
        center_gate = max(0.0, 1.0 - center_ratio * center_ratio)
        forward_distance = (
            min(max(ego.linear_velocity_mps, 0.0), self.config.target_velocity_mps)
            * self.dt_s
        )
        reward_progress = (
            self.config.progress_per_m
            * forward_distance
            * alignment
            * center_gate
        )
        lateral_ratio = ego.lateral_error_m / self.config.lane_lateral_scale_m
        heading_ratio = ego.heading_error_rad / self.config.lane_heading_scale_rad
        reward_lane = (
            self.config.lane_lateral_weight * lateral_ratio * lateral_ratio
            + self.config.lane_heading_weight * heading_ratio * heading_ratio
        )

        yellow_clearance = (
            self.config.lane_center_to_yellow_m
            + ego.lateral_error_m
            - self.config.ego_half_width_m
        )
        warning_fraction = max(
            0.0,
            (self.config.yellow_warning_clearance_m - yellow_clearance)
            / self.config.yellow_warning_clearance_m,
        )
        reward_yellow = self.config.yellow_warning_weight * warning_fraction**2

        linear_change = (
            action.linear_velocity_mps
            - self._previous_action.linear_velocity_mps
        ) / self.config.maximum_linear_velocity_mps
        angular_change = (
            action.angular_velocity_rad_s
            - self._previous_action.angular_velocity_rad_s
        ) / self.config.maximum_angular_velocity_rad_s
        normalized_steering = (
            action.angular_velocity_rad_s
            / self.config.maximum_angular_velocity_rad_s
        )
        reward_comfort = (
            self.config.comfort_linear_change_weight * linear_change**2
            + self.config.comfort_angular_change_weight * angular_change**2
            + self.config.comfort_steering_weight * normalized_steering**2
        )

        invalid_pose = bool(
            simulator_terminated and simulator_done_code == "invalid-pose"
        )
        # The clearance is a subtraction of decimal geometry constants; keep
        # exact contact stable against binary floating-point round-off.
        yellow_contact = yellow_clearance <= 1.0e-9
        (
            yellow_crossing,
            yellow_recovery_started,
            yellow_recovered,
        ) = self._update_yellow_recovery(
            yellow_contact=yellow_contact,
            yellow_clearance_m=yellow_clearance,
            road_curvature_inv_m=road_curvature_inv_m,
        )
        lane_departure = (
            abs(ego.lateral_error_m) > self.config.lane_departure_threshold_m
        )
        heading_error = abs(
            _wrap_angle(world_pose.heading_rad - self._start_pose.heading_rad)
        )
        lap_completed = bool(
            self._left_start
            and self._path_length_m >= self.config.minimum_path_length_m
            and start_distance <= self.config.finish_radius_m
            and heading_error <= self.config.finish_heading_tolerance_rad
            and not self._yellow_recovery_active
            and not yellow_crossing
        )

        terminal_reward = 0.0
        termination_reason: str | None = None
        if invalid_pose:
            terminal_reward = self.config.invalid_pose_penalty
            termination_reason = "invalid_pose"
        elif yellow_crossing:
            terminal_reward = self.config.yellow_crossing_penalty
            termination_reason = (
                "yellow_recovery_failed"
                if self.config.yellow_curve_recovery_enabled
                and self._yellow_recovery_grace_remaining < 0
                else "yellow_crossing"
            )
        elif lane_departure:
            terminal_reward = self.config.lane_departure_penalty
            termination_reason = "lane_departure"
        elif lap_completed:
            terminal_reward = self.config.lap_completion_bonus
            termination_reason = "lap_completed"
        elif simulator_terminated:
            termination_reason = f"simulator:{simulator_done_code}"

        terminated = termination_reason is not None
        truncated = bool(not terminated and (simulator_truncated or horizon_reached))
        truncation_reason = "horizon" if truncated else None
        terms = LaneRewardTerms(
            progress=reward_progress,
            lane=reward_lane,
            yellow=reward_yellow,
            comfort=reward_comfort,
            living=self.config.living_penalty,
            terminal=terminal_reward,
        )
        self._previous_pose = world_pose
        self._previous_action = action
        return LaneStepOutcome(
            reward_terms=terms,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
            truncation_reason=truncation_reason,
            path_length_m=self._path_length_m,
            yellow_clearance_m=yellow_clearance,
            lap_completed=lap_completed,
            yellow_contact=yellow_contact,
            yellow_crossing=yellow_crossing,
            yellow_recovery_started=yellow_recovery_started,
            yellow_recovery_active=self._yellow_recovery_active,
            yellow_recovered=yellow_recovered,
            lane_departure=lane_departure,
            invalid_pose=invalid_pose,
        )

    def _update_yellow_recovery(
        self,
        *,
        yellow_contact: bool,
        yellow_clearance_m: float,
        road_curvature_inv_m: float | None,
    ) -> tuple[bool, bool, bool]:
        """Classify contact while retaining a mandatory post-curve recovery.

        Curvature is privileged reward-side geometry. It never enters the
        policy observation. Legacy protocols keep the original immediate
        terminal behavior because recovery is disabled by default.
        """

        if not self.config.yellow_curve_recovery_enabled:
            return yellow_contact, False, False

        curvature = 0.0 if road_curvature_inv_m is None else road_curvature_inv_m
        on_curve = (
            abs(curvature)
            >= self.config.yellow_curve_min_abs_curvature_inv_m
        )
        deep_crossing = (
            yellow_clearance_m
            < -self.config.yellow_curve_max_penetration_m
        )
        recovery_started = False
        recovered = False

        if yellow_contact:
            self._yellow_recovery_clear_streak = 0
            if deep_crossing:
                return True, False, False
            if on_curve:
                recovery_started = not self._yellow_recovery_active
                self._yellow_recovery_active = True
                self._yellow_recovery_grace_remaining = (
                    self.config.yellow_curve_exit_grace_steps
                )
                return False, recovery_started, False
            if not self._yellow_recovery_active:
                return True, False, False
            self._yellow_recovery_grace_remaining -= 1
            if self._yellow_recovery_grace_remaining < 0:
                return True, False, False
            return False, False, False

        if self._yellow_recovery_active:
            self._yellow_recovery_clear_streak += 1
            if on_curve:
                self._yellow_recovery_grace_remaining = (
                    self.config.yellow_curve_exit_grace_steps
                )
            if (
                self._yellow_recovery_clear_streak
                >= self.config.yellow_recovery_clear_steps
            ):
                self._yellow_recovery_active = False
                self._yellow_recovery_grace_remaining = 0
                self._yellow_recovery_clear_streak = 0
                recovered = True
        return False, recovery_started, recovered


def _wrap_angle(angle: float) -> float:
    return atan2(sin(angle), cos(angle))
