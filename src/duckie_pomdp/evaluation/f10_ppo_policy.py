"""Deterministic evaluation utilities for the staged PPO curriculum."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.control.ppo import PPOAgent
from duckie_pomdp.control.ppo_protocol import PPOCurriculumProtocol


FloatVector = NDArray[np.float32]


class EvaluationPolicy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, observation: FloatVector) -> FloatVector: ...


class PPODeterministicPolicy:
    name = "ppo"

    def __init__(self, agent: PPOAgent) -> None:
        self._agent = agent

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: FloatVector) -> FloatVector:
        return self._agent.act(observation, deterministic=True).environment_action


class RandomPolicy:
    name = "random"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed + 900_001)

    def act(self, observation: FloatVector) -> FloatVector:
        del observation
        return self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


class AlwaysStopPolicy:
    name = "always_stop"

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: FloatVector) -> FloatVector:
        del observation
        return np.asarray((-1.0, 0.0), dtype=np.float32)


class BeliefAwareSimpleController:
    """Non-learning audit baseline using only normalized policy fields."""

    name = "simple_controller"

    def __init__(self, protocol: PPOCurriculumProtocol) -> None:
        self._indices = {
            name: protocol.observation_order.index(name)
            for name in protocol.observation_order
        }
        self._scales = dict(
            zip(protocol.observation_order, protocol.observation_scales, strict=True)
        )
        self._lateral_name, self._heading_name = _lane_policy_fields(protocol)
        self._curvature_name = (
            "lane_curvature_mean_inv_m"
            if "lane_curvature_mean_inv_m" in protocol.observation_order
            else None
        )

    def reset(self, seed: int) -> None:
        del seed

    def _physical(self, observation: FloatVector, name: str) -> float:
        return float(observation[self._indices[name]] * self._scales[name])

    def act(self, observation: FloatVector) -> FloatVector:
        lateral = self._physical(observation, self._lateral_name)
        heading = self._physical(observation, self._heading_name)
        velocity = 0.18
        pedestrian_exists = self._physical(
            observation, "pedestrian_existence_probability"
        )
        pedestrian_range = self._physical(observation, "pedestrian_range_mean_m")
        pedestrian_bearing = abs(
            self._physical(observation, "pedestrian_bearing_mean_rad")
        )
        stop_satisfied = self._physical(observation, "stop_mode_satisfied") >= 0.5
        pedestrian_in_path_corridor = pedestrian_bearing <= 0.65
        # The combined V1 route encounters the single crossing before the stop
        # line. Once the observable stop obligation is satisfied, subsequent
        # Duckie beliefs on this route are detector false alarms over yellow
        # lane dashes, not a second configured crossing. This reference-only
        # rule uses policy-visible state and no privileged geometry.
        if (
            not stop_satisfied
            and pedestrian_exists >= 0.5
            and pedestrian_in_path_corridor
        ):
            if pedestrian_range <= 0.75:
                velocity = 0.0
            elif pedestrian_range <= 1.10:
                velocity = min(velocity, 0.08)

        stop_required = self._physical(observation, "stop_mode_required") >= 0.5
        stop_distance = self._physical(observation, "stop_line_distance_m")
        # Enter the agent-side completion zone (<= 0.18 m) before commanding
        # zero.  Stopping earlier would leave the observable obligation stuck in
        # REQUIRED even if the evaluation-only reward considered the stop safe.
        if stop_required and stop_distance <= 0.15:
            velocity = 0.0
        elif stop_required and stop_distance <= 0.50:
            velocity = min(velocity, 0.08)
        elif stop_satisfied:
            velocity = min(velocity, 0.20)

        curvature = (
            self._physical(observation, self._curvature_name)
            if self._curvature_name is not None
            else 0.0
        )
        feed_forward = velocity * float(np.clip(curvature, -4.0, 4.0))
        omega = float(
            np.clip(2.0 * lateral + 3.0 * heading + feed_forward, -4.0, 4.0)
        )
        turn_fraction = min(abs(omega) / 4.0, 1.0)
        velocity = max(
            velocity * (1.0 - 0.55 * turn_fraction), min(velocity, 0.08)
        )
        return np.asarray((2.0 * velocity / 0.4 - 1.0, omega / 4.0), dtype=np.float32)


@dataclass(frozen=True)
class PPOEpisodeEvaluation:
    stage: str
    policy: str
    checkpoint_step: int | None
    seed: int
    scenario: str
    steps: int
    total_return: float
    completed: bool
    progress_m: float
    collision: bool
    unsafe_proximity_events: int
    minimum_pedestrian_clearance_m: float | None
    stop_completed: bool
    stop_violation: bool
    restarted_after_stop: bool
    lane_failure: bool
    yellow_contact_steps: int
    yellow_recovery_events: int
    yellow_recovery_successes: int
    yellow_recovery_failures: int
    invalid_pose: bool
    timeout: bool
    mean_abs_lateral_error_m: float
    mean_abs_heading_error_rad: float
    mean_v_cmd_mps: float
    mean_abs_omega_cmd_rad_s: float
    mean_action_change: float
    stationary_fraction: float
    reward_progress: float
    reward_lane: float
    reward_pedestrian: float
    reward_stop: float
    reward_smoothness: float
    reward_terminal: float
    termination_reason: str | None
    truncation_reason: str | None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def run_episode(
    environment: object,
    *,
    seed: int,
    policy: EvaluationPolicy,
    protocol: PPOCurriculumProtocol,
    checkpoint_step: int | None = None,
) -> PPOEpisodeEvaluation:
    observation, reset_info = environment.reset(seed=seed)
    policy.reset(seed)
    stage = protocol.stage(str(reset_info["stage"]))
    totals = {
        name: 0.0
        for name in (
            "reward_progress", "reward_lane", "reward_pedestrian",
            "reward_stop", "reward_smoothness", "reward_terminal",
        )
    }
    abs_lateral: list[float] = []
    abs_heading: list[float] = []
    velocities: list[float] = []
    omegas: list[float] = []
    action_changes: list[float] = []
    clearances: list[float] = []
    previous_action = np.zeros(2, dtype=np.float32)
    unsafe_events = 0
    total_return = 0.0
    stop_ever_completed = False
    restarted = False
    yellow_contact_steps = 0
    yellow_recovery_events = 0
    yellow_recovery_successes = 0
    yellow_recovery_failures = 0
    last_info: dict[str, object] | None = None
    for _ in range(stage.episode_horizon_steps):
        action = np.asarray(policy.act(observation), dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("evaluation policy returned an invalid action")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("evaluation action is outside [-1,1]")
        observation, reward, terminated, truncated, info = environment.step(action)
        last_info = info
        evaluation_gt = info["evaluation_gt"]
        abs_lateral.append(abs(float(evaluation_gt["lane_lateral_error_m"])))
        abs_heading.append(abs(float(evaluation_gt["lane_heading_error_rad"])))
        total_return += float(reward)
        for name in totals:
            totals[name] += float(info[name])
        physical = np.asarray((info["v_cmd"], info["omega_cmd"]), dtype=np.float32)
        delta = np.asarray(
            ((physical[0] - previous_action[0]) / 0.4, (physical[1] - previous_action[1]) / 4.0),
            dtype=np.float32,
        )
        action_changes.append(float(np.linalg.norm(delta)))
        previous_action = physical
        velocities.append(float(physical[0]))
        omegas.append(abs(float(physical[1])))
        unsafe_events += int(bool(info["unsafe_proximity"]))
        clearance = info["pedestrian_clearance_m"]
        if clearance is not None and isfinite(float(clearance)):
            clearances.append(float(clearance))
        stop_ever_completed = stop_ever_completed or bool(info["stop_completed"])
        restarted = restarted or (stop_ever_completed and float(physical[0]) > 0.08)
        yellow_contact_steps += int(bool(info["yellow_contact"]))
        yellow_recovery_events += int(bool(info["yellow_recovery_started"]))
        yellow_recovery_successes += int(bool(info["yellow_recovered"]))
        yellow_recovery_failures += int(
            info["termination_reason"] == "yellow_recovery_failed"
        )
        if terminated or truncated:
            break
    if last_info is None:
        raise RuntimeError("evaluation episode produced no step")
    return PPOEpisodeEvaluation(
        stage=stage.key,
        policy=policy.name,
        checkpoint_step=checkpoint_step,
        seed=seed,
        scenario=str(reset_info["scenario"]),
        steps=len(velocities),
        total_return=total_return,
        completed=bool(last_info["completed"]),
        progress_m=float(last_info["progress_m"]),
        collision=bool(last_info["collision"]),
        unsafe_proximity_events=unsafe_events,
        minimum_pedestrian_clearance_m=min(clearances) if clearances else None,
        stop_completed=stop_ever_completed,
        stop_violation=bool(last_info["stop_violation"]),
        restarted_after_stop=restarted,
        lane_failure=bool(last_info["lane_failure"]),
        yellow_contact_steps=yellow_contact_steps,
        yellow_recovery_events=yellow_recovery_events,
        yellow_recovery_successes=yellow_recovery_successes,
        yellow_recovery_failures=yellow_recovery_failures,
        invalid_pose=bool(last_info["invalid_pose"]),
        timeout=bool(last_info["truncation_reason"]),
        mean_abs_lateral_error_m=_mean(abs_lateral),
        mean_abs_heading_error_rad=_mean(abs_heading),
        mean_v_cmd_mps=_mean(velocities),
        mean_abs_omega_cmd_rad_s=_mean(omegas),
        mean_action_change=_mean(action_changes),
        stationary_fraction=float(np.mean(np.asarray(velocities) <= 0.025)),
        **totals,
        termination_reason=_optional(last_info["termination_reason"]),
        truncation_reason=_optional(last_info["truncation_reason"]),
    )


def summarize_episodes(rows: Sequence[PPOEpisodeEvaluation]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty PPO evaluation")
    minima = [row.minimum_pedestrian_clearance_m for row in rows if row.minimum_pedestrian_clearance_m is not None]
    return {
        "episodes": len(rows),
        "completion_rate": _rate(row.completed for row in rows),
        "mean_progress_m": _mean([row.progress_m for row in rows]),
        "mean_return": _mean([row.total_return for row in rows]),
        "collision_rate": _rate(row.collision for row in rows),
        "unsafe_episode_rate": _rate(row.unsafe_proximity_events > 0 for row in rows),
        "minimum_pedestrian_clearance_m": min(minima) if minima else None,
        "stop_completion_rate": _rate(row.stop_completed for row in rows),
        "stop_violation_rate": _rate(row.stop_violation for row in rows),
        "restart_rate": _rate(row.restarted_after_stop for row in rows),
        "lane_failure_rate": _rate(row.lane_failure for row in rows),
        "mean_yellow_contact_steps": _mean([row.yellow_contact_steps for row in rows]),
        "yellow_recovery_event_rate": _rate(row.yellow_recovery_events > 0 for row in rows),
        "yellow_recovery_success_rate": (
            sum(row.yellow_recovery_successes for row in rows)
            / max(sum(row.yellow_recovery_events for row in rows), 1)
        ),
        "yellow_recovery_failure_rate": _rate(
            row.yellow_recovery_failures > 0 for row in rows
        ),
        "invalid_pose_rate": _rate(row.invalid_pose for row in rows),
        "timeout_rate": _rate(row.timeout for row in rows),
        "mean_abs_lateral_error_m": _mean([row.mean_abs_lateral_error_m for row in rows]),
        "mean_abs_heading_error_rad": _mean([row.mean_abs_heading_error_rad for row in rows]),
        "mean_v_cmd_mps": _mean([row.mean_v_cmd_mps for row in rows]),
        "mean_abs_omega_cmd_rad_s": _mean([row.mean_abs_omega_cmd_rad_s for row in rows]),
        "mean_action_change": _mean([row.mean_action_change for row in rows]),
        "stationary_fraction": _mean([row.stationary_fraction for row in rows]),
        **{
            f"mean_{name}": _mean([float(getattr(row, name)) for row in rows])
            for name in (
                "reward_progress", "reward_lane", "reward_pedestrian",
                "reward_stop", "reward_smoothness", "reward_terminal",
            )
        },
    }


def _lane_policy_fields(protocol: PPOCurriculumProtocol) -> tuple[str, str]:
    if "lane_lateral_error_mean_m" in protocol.observation_order:
        return "lane_lateral_error_mean_m", "lane_heading_error_mean_rad"
    return "lateral_error_m", "heading_error_rad"


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _rate(values) -> float:
    materialized = tuple(bool(value) for value in values)
    return sum(materialized) / len(materialized)


def _optional(value: object) -> str | None:
    return None if value is None else str(value)
