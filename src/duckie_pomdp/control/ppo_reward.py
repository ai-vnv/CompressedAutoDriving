"""One six-component reward interface across the F10-PPO curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import SensorObservation
from duckie_pomdp.domain.privileged import PrivilegedSimulatorState, WorldPose

from .lane_reward import LaneRewardConfig, LaneRewardEvaluator
from .ppo_protocol import CurriculumStage, PPOCurriculumProtocol
from .reward import (
    F10RewardConfig,
    F10RewardEvaluator,
    LoopHazardRewardEvaluator,
)


@dataclass(frozen=True)
class PPORewardTerms:
    progress: float
    lane: float
    pedestrian: float
    stop: float
    smoothness: float
    terminal: float

    @property
    def total(self) -> float:
        values = (
            self.progress,
            self.lane,
            self.pedestrian,
            self.stop,
            self.smoothness,
            self.terminal,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("PPO reward terms must be finite")
        return sum(values)


@dataclass(frozen=True)
class PPORewardOutcome:
    terms: PPORewardTerms
    terminated: bool
    truncated: bool
    termination_reason: str | None
    truncation_reason: str | None
    progress_m: float
    completed: bool
    lane_failure: bool
    yellow_contact: bool
    yellow_recovery_started: bool
    yellow_recovery_active: bool
    yellow_recovered: bool
    invalid_pose: bool
    collision: bool
    unsafe_proximity: bool
    pedestrian_clearance_m: float | None
    stop_completed: bool
    stop_violation: bool

    @property
    def reward(self) -> float:
        return self.terms.total


class PPORewardEvaluator:
    """Delegates map geometry while preserving one public decomposition."""

    def __init__(
        self,
        protocol: PPOCurriculumProtocol,
        stage: CurriculumStage,
        *,
        dt_s: float,
        route_heading_rad: float | None,
    ) -> None:
        self.stage = stage
        reward = protocol.raw["reward"]
        reward_config = F10RewardConfig(
            progress_per_m=float(reward["progress_per_m"]),
            living_penalty=float(reward["living_penalty"]),
            lane_lateral_weight=float(reward["lane_lateral_weight"]),
            lane_lateral_scale_m=float(reward["lane_lateral_scale_m"]),
            lane_heading_weight=float(reward["lane_heading_weight"]),
            lane_heading_scale_rad=float(reward["lane_heading_scale_rad"]),
            lane_departure_threshold_m=float(reward["lane_departure_threshold_m"]),
            pedestrian_safety_distance_m=float(reward["pedestrian_safety_distance_m"]),
            pedestrian_unsafe_distance_m=float(reward["pedestrian_unsafe_distance_m"]),
            pedestrian_collision_distance_m=float(reward["pedestrian_collision_distance_m"]),
            pedestrian_proximity_weight=float(reward["pedestrian_proximity_weight"]),
            pedestrian_unsafe_penalty=float(reward["pedestrian_unsafe_penalty"]),
            comfort_linear_change_weight=float(reward["smooth_linear_change_weight"]),
            comfort_angular_change_weight=float(reward["smooth_angular_change_weight"]),
            comfort_steering_weight=float(reward["smooth_steering_weight"]),
            stop_approach_distance_m=float(reward["stop_approach_distance_m"]),
            stop_speed_threshold_mps=float(reward["stop_speed_threshold_mps"]),
            stop_hold_steps=int(reward["stop_hold_steps"]),
            stop_approach_speed_weight=float(reward["stop_approach_speed_weight"]),
            stop_completion_bonus=float(reward["stop_completion_bonus"]),
            stop_violation_penalty=float(reward["stop_violation_penalty"]),
            pedestrian_collision_terminal_penalty=float(reward["pedestrian_collision_terminal_penalty"]),
            invalid_pose_terminal_penalty=float(reward["invalid_pose_penalty"]),
            success_terminal_bonus=float(reward["route_completion_bonus"]),
            success_progress_m=float(reward["route_success_progress_m"]),
            maximum_linear_velocity_mps=protocol.action_bounds[1],
            maximum_angular_velocity_rad_s=protocol.action_bounds[3],
            lane_departure_terminal_penalty=float(reward["lane_failure_penalty"]),
            pedestrian_stationary_proximity_factor=float(
                reward.get("pedestrian_stationary_proximity_factor", 0.25)
            ),
            pedestrian_safe_clearance_bonus=float(
                reward.get("pedestrian_safe_clearance_bonus", 0.0)
            ),
        )
        if stage.map_name in {"small_loop", "experiment_loop"}:
            minimum_path = 4.50 if stage.map_name == "small_loop" else 6.80
            self._lane = LaneRewardEvaluator(
                LaneRewardConfig(
                    target_velocity_mps=float(reward["target_velocity_mps"]),
                    progress_per_m=float(reward["progress_per_m"]),
                    living_penalty=float(reward["living_penalty"]),
                    lane_lateral_weight=float(reward["lane_lateral_weight"]),
                    lane_lateral_scale_m=float(reward["lane_lateral_scale_m"]),
                    lane_heading_weight=float(reward["lane_heading_weight"]),
                    lane_heading_scale_rad=float(reward["lane_heading_scale_rad"]),
                    yellow_warning_weight=float(reward["yellow_warning_weight"]),
                    comfort_linear_change_weight=float(reward["smooth_linear_change_weight"]),
                    comfort_angular_change_weight=float(reward["smooth_angular_change_weight"]),
                    comfort_steering_weight=float(reward["smooth_steering_weight"]),
                    lap_completion_bonus=float(reward["lane_completion_bonus"]),
                    yellow_crossing_penalty=float(reward["lane_failure_penalty"]),
                    lane_departure_penalty=float(reward["lane_failure_penalty"]),
                    invalid_pose_penalty=float(reward["invalid_pose_penalty"]),
                    lane_center_to_yellow_m=float(reward["lane_center_to_yellow_m"]),
                    ego_half_width_m=float(reward["ego_half_width_m"]),
                    yellow_warning_clearance_m=float(reward["yellow_warning_clearance_m"]),
                    lane_departure_threshold_m=float(reward["lane_departure_threshold_m"]),
                    minimum_path_length_m=minimum_path,
                    leave_start_radius_m=0.35,
                    finish_radius_m=0.11,
                    finish_heading_tolerance_rad=0.35,
                    maximum_linear_velocity_mps=protocol.action_bounds[1],
                    maximum_angular_velocity_rad_s=protocol.action_bounds[3],
                    yellow_curve_recovery_enabled=bool(
                        reward.get("yellow_curve_recovery_enabled", False)
                    ),
                    yellow_curve_min_abs_curvature_inv_m=float(
                        reward.get("yellow_curve_min_abs_curvature_inv_m", 0.75)
                    ),
                    yellow_curve_max_penetration_m=float(
                        reward.get("yellow_curve_max_penetration_m", 0.035)
                    ),
                    yellow_curve_exit_grace_steps=int(
                        reward.get("yellow_curve_exit_grace_steps", 15)
                    ),
                    yellow_recovery_clear_steps=int(
                        reward.get("yellow_recovery_clear_steps", 3)
                    ),
                ),
                dt_s=dt_s,
            )
            self._pomdp = None
            self._hazard = (
                LoopHazardRewardEvaluator(
                    reward_config,
                    pedestrian_active=stage.pedestrian_active,
                    stop_active=stage.stop_active,
                )
                if stage.pedestrian_active or stage.stop_active
                else None
            )
        else:
            if route_heading_rad is None:
                raise ValueError("POMDP reward requires route heading")
            self._lane = None
            self._pomdp = F10RewardEvaluator(
                reward_config,
                route_heading_rad=route_heading_rad,
                pedestrian_active=stage.pedestrian_active,
                stop_active=stage.stop_active,
            )
            self._hazard = None

    def reset(
        self,
        observation: SensorObservation,
        privileged: PrivilegedSimulatorState,
    ) -> None:
        if self._lane is not None:
            self._lane.reset(privileged.ego_world_pose)
            if self._hazard is not None:
                self._hazard.reset(privileged)
        else:
            assert self._pomdp is not None
            self._pomdp.reset(observation, privileged)

    def evaluate(
        self,
        *,
        action: PolicyAction,
        observation: SensorObservation,
        privileged: PrivilegedSimulatorState,
        world_pose: WorldPose,
        simulator_terminated: bool,
        simulator_truncated: bool,
        simulator_done_code: str,
        horizon_reached: bool,
    ) -> PPORewardOutcome:
        if self._lane is not None:
            value = self._lane.evaluate(
                action=action,
                observation=observation,
                world_pose=world_pose,
                simulator_terminated=simulator_terminated,
                simulator_truncated=simulator_truncated,
                simulator_done_code=simulator_done_code,
                horizon_reached=horizon_reached,
                road_curvature_inv_m=(
                    privileged.true_pomdp_state.road.curvature_inv_m
                ),
            )
            hazard = (
                None
                if self._hazard is None
                else self._hazard.evaluate(
                    observation=observation,
                    privileged=privileged,
                )
            )
            terminated = value.terminated or bool(
                hazard is not None and hazard.terminated
            )
            termination_reason = (
                hazard.termination_reason
                if hazard is not None and hazard.terminated
                else value.termination_reason
            )
            terms = PPORewardTerms(
                progress=value.reward_terms.progress + value.reward_terms.living,
                lane=value.reward_terms.lane + value.reward_terms.yellow,
                pedestrian=(0.0 if hazard is None else hazard.pedestrian_reward),
                stop=(0.0 if hazard is None else hazard.stop_reward),
                smoothness=value.reward_terms.comfort,
                terminal=value.reward_terms.terminal
                + (0.0 if hazard is None else hazard.terminal_reward),
            )
            return PPORewardOutcome(
                terms=terms,
                terminated=terminated,
                truncated=value.truncated,
                termination_reason=termination_reason,
                truncation_reason=value.truncation_reason,
                progress_m=value.path_length_m,
                completed=value.lap_completed,
                lane_failure=value.yellow_crossing or value.lane_departure,
                yellow_contact=value.yellow_contact,
                yellow_recovery_started=value.yellow_recovery_started,
                yellow_recovery_active=value.yellow_recovery_active,
                yellow_recovered=value.yellow_recovered,
                invalid_pose=value.invalid_pose,
                collision=(False if hazard is None else hazard.pedestrian_collision),
                unsafe_proximity=(False if hazard is None else hazard.unsafe_proximity),
                pedestrian_clearance_m=(
                    None if hazard is None else hazard.pedestrian_clearance_m
                ),
                stop_completed=(False if hazard is None else hazard.stop_completed),
                stop_violation=(False if hazard is None else hazard.stop_violation),
            )

        assert self._pomdp is not None
        value = self._pomdp.evaluate(
            action=action,
            observation=observation,
            privileged=privileged,
            simulator_terminated=simulator_terminated,
            simulator_truncated=simulator_truncated,
            simulator_done_code=simulator_done_code,
            horizon_reached=horizon_reached,
        )
        return PPORewardOutcome(
            terms=PPORewardTerms(
                progress=value.reward_terms.progress,
                lane=value.reward_terms.lane,
                pedestrian=value.reward_terms.pedestrian,
                stop=value.reward_terms.stop,
                smoothness=value.reward_terms.comfort,
                terminal=value.reward_terms.terminal,
            ),
            terminated=value.terminated,
            truncated=value.truncated,
            termination_reason=value.termination_reason,
            truncation_reason=value.truncation_reason,
            progress_m=value.progress_m,
            completed=value.termination_reason == "success",
            lane_failure=value.lane_departure,
            yellow_contact=False,
            yellow_recovery_started=False,
            yellow_recovery_active=False,
            yellow_recovered=False,
            invalid_pose=value.invalid_pose,
            collision=value.pedestrian_collision,
            unsafe_proximity=value.unsafe_proximity,
            pedestrian_clearance_m=value.pedestrian_clearance_m,
            stop_completed=value.stop_completed,
            stop_violation=value.stop_violation,
        )
