"""Deterministic F10 policy evaluation and safety-first checkpoint selection.

This module consumes only the Gym-compatible agent observation and the
evaluation fields returned by :class:`F10GymEnvironment`.  It never reaches
through the environment to simulator privileged state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.control.f10_protocol import F10Protocol
from duckie_pomdp.control.sac import SACAgent


FloatVector = NDArray[np.float32]


class EvaluationPolicy(Protocol):
    """Small runtime boundary shared by SAC and deterministic baselines."""

    name: str

    def reset(self, seed: int) -> None:
        """Reset episode-local state without reading simulator truth."""

    def act(self, observation: FloatVector) -> FloatVector:
        """Return one normalized action in ``[-1, 1]^2``."""


class SACDeterministicPolicy:
    name = "sac"

    def __init__(self, agent: SACAgent) -> None:
        self._agent = agent

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: FloatVector) -> FloatVector:
        return self._agent.act(observation, deterministic=True)


class RandomPolicy:
    name = "random"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed + 700_001)

    def act(self, observation: FloatVector) -> FloatVector:
        del observation
        return self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


class AlwaysStopPolicy:
    name = "always_stop"

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: FloatVector) -> FloatVector:
        del observation
        # The F10 affine action map sends a_v=-1 to 0 m/s.
        return np.asarray((-1.0, 0.0), dtype=np.float32)


class SimpleControllerPolicy:
    """Agent-visible proportional lane/stop controller used by the audit."""

    name = "simple_controller"

    def __init__(self, protocol: F10Protocol) -> None:
        self._indices = {
            name: protocol.observation_order.index(name)
            for name in (
                "lateral_error_m",
                "heading_error_rad",
                "actual_linear_velocity_mps",
                "stop_line_distance_m",
            )
        }
        self._scales = dict(
            zip(protocol.observation_order, protocol.observation_scales, strict=True)
        )
        self._hold_steps = 0
        self._stop_completed = False

    def reset(self, seed: int) -> None:
        del seed
        self._hold_steps = 0
        self._stop_completed = False

    def _physical(self, observation: FloatVector, name: str) -> float:
        return float(observation[self._indices[name]] * self._scales[name])

    def act(self, observation: FloatVector) -> FloatVector:
        lateral = self._physical(observation, "lateral_error_m")
        heading = self._physical(observation, "heading_error_rad")
        actual_velocity = self._physical(
            observation, "actual_linear_velocity_mps"
        )
        stop_distance = self._physical(observation, "stop_line_distance_m")
        if not self._stop_completed and 0.0 <= stop_distance <= 0.24:
            if actual_velocity <= 0.025:
                self._hold_steps += 1
            if self._hold_steps >= 15:
                self._stop_completed = True
            velocity = 0.0
        elif not self._stop_completed and stop_distance <= 0.40:
            velocity = 0.08
        else:
            velocity = 0.28
        omega = float(np.clip(-1.5 * lateral - 0.8 * heading, -0.8, 0.8))
        return np.asarray((2.0 * velocity / 0.4 - 1.0, omega / 4.0), dtype=np.float32)


@dataclass(frozen=True)
class EpisodeEvaluation:
    policy: str
    checkpoint_step: int | None
    seed: int
    scenario: str
    steps: int
    total_return: float
    success: bool
    progress_m: float
    collision: bool
    invalid_pose: bool
    timeout: bool
    stop_completed: bool
    stop_violation: bool
    lane_departure_events: int
    unsafe_proximity_events: int
    minimum_pedestrian_clearance_m: float | None
    mean_abs_lateral_error_m: float
    mean_abs_heading_error_rad: float
    mean_v_cmd_mps: float
    mean_abs_omega_cmd_rad_s: float
    mean_action_change: float
    steering_oscillations: int
    safety_region_steps: int
    clear_region_steps: int
    mean_v_cmd_safety_region_mps: float | None
    mean_v_cmd_clear_region_mps: float | None
    reward_progress: float
    reward_lane: float
    reward_stop: float
    reward_pedestrian: float
    reward_comfort: float
    reward_terminal: float
    termination_reason: str | None
    truncation_reason: str | None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointScore:
    path: Path
    global_step: int
    checkpoint_sha256: str
    summary: dict[str, object]


@dataclass(frozen=True)
class CheckpointSelection:
    selected: CheckpointScore
    best_return: CheckpointScore
    last: CheckpointScore
    safety_filter_passed: bool
    selection_reason: str


def run_episode(
    environment: object,
    *,
    seed: int,
    policy: EvaluationPolicy,
    protocol: F10Protocol,
    checkpoint_step: int | None = None,
) -> EpisodeEvaluation:
    """Evaluate one episode through the public Gym boundary."""

    observation, reset_info = environment.reset(seed=seed)
    policy.reset(seed)
    scales = dict(
        zip(protocol.observation_order, protocol.observation_scales, strict=True)
    )
    lateral_index = protocol.observation_order.index("lateral_error_m")
    heading_index = protocol.observation_order.index("heading_error_rad")
    safety_distance = float(
        protocol.raw["reward"]["pedestrian_safety_distance_m"]
    )

    totals = {
        name: 0.0
        for name in (
            "reward_progress",
            "reward_lane",
            "reward_stop",
            "reward_pedestrian",
            "reward_comfort",
            "reward_terminal",
        )
    }
    abs_lateral: list[float] = []
    abs_heading: list[float] = []
    velocities: list[float] = []
    abs_omegas: list[float] = []
    action_changes: list[float] = []
    clear_velocities: list[float] = []
    safety_velocities: list[float] = []
    clearances: list[float] = []
    previous_physical = np.zeros(2, dtype=np.float32)
    previous_turn_sign = 0
    oscillations = 0
    lane_departures = 0
    unsafe_events = 0
    total_return = 0.0
    last_info: dict[str, object] | None = None

    horizon = int(protocol.raw["simulator"]["episode_horizon_steps"])
    for _ in range(horizon):
        abs_lateral.append(
            abs(float(observation[lateral_index]) * scales["lateral_error_m"])
        )
        abs_heading.append(
            abs(float(observation[heading_index]) * scales["heading_error_rad"])
        )
        normalized_action = np.asarray(policy.act(observation), dtype=np.float32)
        if normalized_action.shape != (2,) or not np.all(np.isfinite(normalized_action)):
            raise ValueError("evaluation policy returned an invalid action")
        if np.any(normalized_action < -1.0) or np.any(normalized_action > 1.0):
            raise ValueError("evaluation policy returned an out-of-domain action")
        observation, reward, terminated, truncated, info = environment.step(
            normalized_action
        )
        last_info = info
        total_return += float(reward)
        for name in totals:
            totals[name] += float(info[name])

        physical = np.asarray((info["v_cmd"], info["omega_cmd"]), dtype=np.float32)
        normalized_change = np.asarray(
            (
                (physical[0] - previous_physical[0]) / 0.4,
                (physical[1] - previous_physical[1]) / 4.0,
            ),
            dtype=np.float32,
        )
        action_changes.append(float(np.linalg.norm(normalized_change)))
        previous_physical = physical
        velocity = float(info["v_cmd"])
        omega = float(info["omega_cmd"])
        velocities.append(velocity)
        abs_omegas.append(abs(omega))
        turn_sign = 1 if omega > 0.10 else -1 if omega < -0.10 else 0
        if turn_sign and previous_turn_sign and turn_sign != previous_turn_sign:
            oscillations += 1
        if turn_sign:
            previous_turn_sign = turn_sign

        lane_departures += int(bool(info["lane_departure"]))
        unsafe_events += int(bool(info["unsafe_proximity"]))
        clearance = info["pedestrian_clearance_m"]
        if clearance is not None and isfinite(float(clearance)):
            clearance_value = float(clearance)
            clearances.append(clearance_value)
            if clearance_value < safety_distance:
                safety_velocities.append(velocity)
            else:
                clear_velocities.append(velocity)
        if terminated or truncated:
            break

    if last_info is None:  # pragma: no cover - positive horizon is a config invariant
        raise RuntimeError("evaluation episode produced no transition")
    return EpisodeEvaluation(
        policy=policy.name,
        checkpoint_step=checkpoint_step,
        seed=seed,
        scenario=str(reset_info["scenario"]),
        steps=len(velocities),
        total_return=total_return,
        success=last_info["termination_reason"] == "success",
        progress_m=float(last_info["progress_m"]),
        collision=bool(last_info["collision"]),
        invalid_pose=bool(last_info["invalid_pose"]),
        timeout=bool(last_info["truncation_reason"]),
        stop_completed=bool(last_info["stop_completed"]),
        stop_violation=bool(last_info["stop_violation"]),
        lane_departure_events=lane_departures,
        unsafe_proximity_events=unsafe_events,
        minimum_pedestrian_clearance_m=min(clearances) if clearances else None,
        mean_abs_lateral_error_m=_mean(abs_lateral),
        mean_abs_heading_error_rad=_mean(abs_heading),
        mean_v_cmd_mps=_mean(velocities),
        mean_abs_omega_cmd_rad_s=_mean(abs_omegas),
        mean_action_change=_mean(action_changes),
        steering_oscillations=oscillations,
        safety_region_steps=len(safety_velocities),
        clear_region_steps=len(clear_velocities),
        mean_v_cmd_safety_region_mps=_optional_mean(safety_velocities),
        mean_v_cmd_clear_region_mps=_optional_mean(clear_velocities),
        **totals,
        termination_reason=_optional_text(last_info["termination_reason"]),
        truncation_reason=_optional_text(last_info["truncation_reason"]),
    )


def summarize_episodes(rows: Sequence[EpisodeEvaluation]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    minima = [
        row.minimum_pedestrian_clearance_m
        for row in rows
        if row.minimum_pedestrian_clearance_m is not None
    ]
    safety_velocity_sum = sum(
        float(row.mean_v_cmd_safety_region_mps) * row.safety_region_steps
        for row in rows
        if row.mean_v_cmd_safety_region_mps is not None
    )
    clear_velocity_sum = sum(
        float(row.mean_v_cmd_clear_region_mps) * row.clear_region_steps
        for row in rows
        if row.mean_v_cmd_clear_region_mps is not None
    )
    safety_steps = sum(row.safety_region_steps for row in rows)
    clear_steps = sum(row.clear_region_steps for row in rows)
    safety_mean = safety_velocity_sum / safety_steps if safety_steps else None
    clear_mean = clear_velocity_sum / clear_steps if clear_steps else None
    speed_response = (
        clear_mean - safety_mean
        if clear_mean is not None and safety_mean is not None
        else None
    )
    summary: dict[str, object] = {
        "episodes": len(rows),
        "success_rate": _rate(row.success for row in rows),
        "mean_progress_m": _mean([row.progress_m for row in rows]),
        "mean_return": _mean([row.total_return for row in rows]),
        "mean_episode_length": _mean([float(row.steps) for row in rows]),
        "collision_rate": _rate(row.collision for row in rows),
        "invalid_pose_rate": _rate(row.invalid_pose for row in rows),
        "timeout_rate": _rate(row.timeout for row in rows),
        "stop_completion_rate": _rate(row.stop_completed for row in rows),
        "stop_violation_rate": _rate(row.stop_violation for row in rows),
        "lane_departure_episode_rate": _rate(
            row.lane_departure_events > 0 for row in rows
        ),
        "mean_lane_departure_events": _mean(
            [float(row.lane_departure_events) for row in rows]
        ),
        "unsafe_proximity_episode_rate": _rate(
            row.unsafe_proximity_events > 0 for row in rows
        ),
        "minimum_pedestrian_clearance_m": min(minima) if minima else None,
        "mean_abs_lateral_error_m": _mean(
            [row.mean_abs_lateral_error_m for row in rows]
        ),
        "mean_abs_heading_error_rad": _mean(
            [row.mean_abs_heading_error_rad for row in rows]
        ),
        "mean_v_cmd_mps": _mean([row.mean_v_cmd_mps for row in rows]),
        "mean_abs_omega_cmd_rad_s": _mean(
            [row.mean_abs_omega_cmd_rad_s for row in rows]
        ),
        "mean_action_change": _mean([row.mean_action_change for row in rows]),
        "mean_steering_oscillations": _mean(
            [float(row.steering_oscillations) for row in rows]
        ),
        "safety_region_steps": safety_steps,
        "clear_region_steps": clear_steps,
        "mean_v_cmd_safety_region_mps": safety_mean,
        "mean_v_cmd_clear_region_mps": clear_mean,
        "pedestrian_speed_response_mps": speed_response,
    }
    for reward_name in (
        "reward_progress",
        "reward_lane",
        "reward_stop",
        "reward_pedestrian",
        "reward_comfort",
        "reward_terminal",
    ):
        summary[f"mean_{reward_name}"] = _mean(
            [float(getattr(row, reward_name)) for row in rows]
        )
    summary["by_scenario"] = {
        scenario: summarize_episodes(
            [row for row in rows if row.scenario == scenario]
        )
        for scenario in sorted({row.scenario for row in rows})
    } if len({row.scenario for row in rows}) > 1 else {}
    return summary


def select_checkpoint(
    scores: Sequence[CheckpointScore],
    *,
    maximum_collision_rate: float,
    maximum_invalid_pose_rate: float,
) -> CheckpointSelection:
    """Apply the predeclared safety filter before competence ranking."""

    if not scores:
        raise ValueError("checkpoint selection requires at least one candidate")
    ordered = sorted(scores, key=lambda item: item.global_step)
    last = ordered[-1]
    best_return = max(
        ordered,
        key=lambda item: (
            _metric(item, "mean_return"),
            _metric(item, "success_rate"),
            _metric(item, "mean_progress_m"),
            item.global_step,
        ),
    )
    eligible = [
        item
        for item in ordered
        if _metric(item, "collision_rate") <= maximum_collision_rate
        and _metric(item, "invalid_pose_rate") <= maximum_invalid_pose_rate
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                _metric(item, "success_rate"),
                _metric(item, "mean_progress_m"),
                _metric(item, "mean_return"),
                item.global_step,
            ),
        )
        return CheckpointSelection(
            selected=selected,
            best_return=best_return,
            last=last,
            safety_filter_passed=True,
            selection_reason=(
                "passed safety thresholds; ranked by success, progress, return, "
                "then latest step"
            ),
        )
    selected = min(
        ordered,
        key=lambda item: (
            _metric(item, "collision_rate"),
            _metric(item, "invalid_pose_rate"),
            -_metric(item, "success_rate"),
            -_metric(item, "mean_progress_m"),
            -_metric(item, "mean_return"),
            -item.global_step,
        ),
    )
    return CheckpointSelection(
        selected=selected,
        best_return=best_return,
        last=last,
        safety_filter_passed=False,
        selection_reason=(
            "no checkpoint passed safety thresholds; least-unsafe diagnostic "
            "candidate selected and F10 is capped at LIMITED"
        ),
    )


def acceptance_checks(
    sac: dict[str, object],
    random: dict[str, object],
    always_stop: dict[str, object],
    protocol: F10Protocol,
) -> dict[str, bool]:
    config = protocol.raw["acceptance"]
    random_lane_rate = float(random["lane_departure_episode_rate"])
    lane_limit = (
        random_lane_rate
        * float(config["maximum_lane_departure_rate_ratio_vs_random"])
    )
    speed_response = sac["pedestrian_speed_response_mps"]
    return {
        "minimum_success_rate": float(sac["success_rate"])
        >= float(config["minimum_success_rate"]),
        "minimum_mean_progress": float(sac["mean_progress_m"])
        >= float(config["minimum_mean_progress_m"]),
        "progress_gain_over_always_stop": (
            float(sac["mean_progress_m"]) - float(always_stop["mean_progress_m"])
        )
        >= float(config["minimum_progress_gain_over_always_stop_m"]),
        "progress_gain_over_random": (
            float(sac["mean_progress_m"]) - float(random["mean_progress_m"])
        )
        >= float(config["minimum_progress_gain_over_random_m"]),
        "maximum_collision_rate": float(sac["collision_rate"])
        <= float(config["maximum_collision_rate"]),
        "lane_stability_vs_random": float(sac["lane_departure_episode_rate"])
        <= lane_limit,
        "pedestrian_speed_response": speed_response is not None
        and float(speed_response)
        >= float(config["minimum_pedestrian_speed_response_mps"]),
    }


def _metric(item: CheckpointScore, name: str) -> float:
    value = item.summary[name]
    if not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"checkpoint metric {name} must be finite")
    return float(value)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _optional_mean(values: Sequence[float]) -> float | None:
    return _mean(values) if values else None


def _rate(values: Sequence[bool] | object) -> float:
    materialized = list(values)
    return _mean([float(value) for value in materialized])


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)
