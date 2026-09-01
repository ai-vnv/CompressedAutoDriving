"""F10-L1 deterministic evaluation and safety-first checkpoint selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.control.lane_protocol import LaneProtocol
from duckie_pomdp.control.sac import SACAgent


FloatVector = NDArray[np.float32]


class LaneEvaluationPolicy(Protocol):
    name: str

    def reset(self, seed: int) -> None:
        """Reset policy episode state."""

    def act(self, observation: FloatVector) -> FloatVector:
        """Return one normalized action in [-1, 1]^2."""


class LaneSACPolicy:
    name = "sac"

    def __init__(self, agent: SACAgent) -> None:
        self._agent = agent

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: FloatVector) -> FloatVector:
        return self._agent.act(observation, deterministic=True)


class LaneRandomPolicy:
    name = "random"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed + 900_001)

    def act(self, observation: FloatVector) -> FloatVector:
        del observation
        return self._rng.uniform(-1.0, 1.0, 2).astype(np.float32)


class LaneAlwaysStopPolicy:
    name = "always_stop"

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: FloatVector) -> FloatVector:
        del observation
        return np.asarray((-1.0, 0.0), dtype=np.float32)


class LaneSimpleControllerPolicy:
    """Reference P-controller using the same policy observation boundary."""

    name = "simple_controller"

    def __init__(self, protocol: LaneProtocol) -> None:
        self._indices = {
            name: protocol.observation_order.index(name)
            for name in ("lateral_error_m", "heading_error_rad")
        }
        self._scales = dict(
            zip(protocol.observation_order, protocol.observation_scales, strict=True)
        )

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: FloatVector) -> FloatVector:
        lateral = float(
            observation[self._indices["lateral_error_m"]]
            * self._scales["lateral_error_m"]
        )
        heading = float(
            observation[self._indices["heading_error_rad"]]
            * self._scales["heading_error_rad"]
        )
        omega = float(np.clip(5.0 * lateral + 3.0 * heading, -4.0, 4.0))
        turn_fraction = min(abs(omega) / 4.0, 1.0)
        velocity = max(0.20 * (1.0 - 0.45 * turn_fraction), 0.10)
        return np.asarray((2.0 * velocity / 0.4 - 1.0, omega / 4.0), dtype=np.float32)


@dataclass(frozen=True)
class LaneEpisodeEvaluation:
    policy: str
    checkpoint_step: int | None
    seed: int
    steps: int
    total_return: float
    lap_completed: bool
    path_length_m: float
    invalid_pose: bool
    yellow_crossing: bool
    lane_departure: bool
    timeout: bool
    mean_abs_lateral_error_m: float
    p95_abs_lateral_error_m: float
    mean_abs_heading_error_rad: float
    p95_abs_heading_error_rad: float
    minimum_yellow_clearance_m: float
    mean_actual_velocity_mps: float
    mean_v_cmd_mps: float
    mean_abs_omega_cmd_rad_s: float
    mean_action_change: float
    steering_oscillations: int
    reward_progress: float
    reward_lane: float
    reward_yellow: float
    reward_comfort: float
    reward_living: float
    reward_terminal: float
    termination_reason: str | None
    truncation_reason: str | None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LaneCheckpointScore:
    path: Path
    global_step: int
    sha256: str
    summary: dict[str, object]


@dataclass(frozen=True)
class LaneCheckpointSelection:
    selected: LaneCheckpointScore
    best_return: LaneCheckpointScore
    last: LaneCheckpointScore
    safety_filter_passed: bool
    reason: str


def run_lane_episode(
    environment: object,
    *,
    seed: int,
    policy: LaneEvaluationPolicy,
    protocol: LaneProtocol,
    checkpoint_step: int | None = None,
) -> LaneEpisodeEvaluation:
    observation, _ = environment.reset(seed=seed)
    policy.reset(seed)
    lateral: list[float] = []
    heading: list[float] = []
    yellow_clearance: list[float] = []
    actual_velocity: list[float] = []
    velocity_commands: list[float] = []
    angular_commands: list[float] = []
    action_changes: list[float] = []
    totals = {
        name: 0.0
        for name in (
            "reward_progress",
            "reward_lane",
            "reward_yellow",
            "reward_comfort",
            "reward_living",
            "reward_terminal",
        )
    }
    previous_action = np.zeros(2, dtype=np.float32)
    previous_turn_sign = 0
    oscillations = 0
    total_return = 0.0
    last_info: dict[str, object] | None = None
    horizon = int(protocol.raw["simulator"]["episode_horizon_steps"])
    for _ in range(horizon):
        normalized_action = np.asarray(policy.act(observation), dtype=np.float32)
        if normalized_action.shape != (2,) or not np.all(np.isfinite(normalized_action)):
            raise ValueError("lane evaluation policy returned an invalid action")
        if np.any(normalized_action < -1.0) or np.any(normalized_action > 1.0):
            raise ValueError("lane evaluation action is outside [-1, 1]")
        observation, reward, terminated, truncated, info = environment.step(
            normalized_action
        )
        last_info = info
        total_return += float(reward)
        for name in totals:
            totals[name] += float(info[name])
        lateral.append(abs(float(info["lateral_error_m"])))
        heading.append(abs(float(info["heading_error_rad"])))
        yellow_clearance.append(float(info["yellow_clearance_m"]))
        actual_velocity.append(float(info["v_actual"]))
        physical = np.asarray((info["v_cmd"], info["omega_cmd"]), dtype=np.float32)
        velocity_commands.append(float(physical[0]))
        angular_commands.append(float(physical[1]))
        action_changes.append(
            float(
                np.linalg.norm(
                    (
                        (physical - previous_action)
                        / np.asarray((0.4, 4.0), dtype=np.float32)
                    )
                )
            )
        )
        previous_action = physical
        turn_sign = 1 if physical[1] > 0.10 else -1 if physical[1] < -0.10 else 0
        if turn_sign and previous_turn_sign and turn_sign != previous_turn_sign:
            oscillations += 1
        if turn_sign:
            previous_turn_sign = turn_sign
        if terminated or truncated:
            break
    if last_info is None:
        raise RuntimeError("lane evaluation produced no transition")
    return LaneEpisodeEvaluation(
        policy=policy.name,
        checkpoint_step=checkpoint_step,
        seed=seed,
        steps=len(lateral),
        total_return=total_return,
        lap_completed=bool(last_info["lap_completed"]),
        path_length_m=float(last_info["path_length_m"]),
        invalid_pose=bool(last_info["invalid_pose"]),
        yellow_crossing=bool(last_info["yellow_crossing"]),
        lane_departure=bool(last_info["lane_departure"]),
        timeout=bool(last_info["truncation_reason"]),
        mean_abs_lateral_error_m=_mean(lateral),
        p95_abs_lateral_error_m=float(np.percentile(lateral, 95)),
        mean_abs_heading_error_rad=_mean(heading),
        p95_abs_heading_error_rad=float(np.percentile(heading, 95)),
        minimum_yellow_clearance_m=min(yellow_clearance),
        mean_actual_velocity_mps=_mean(actual_velocity),
        mean_v_cmd_mps=_mean(velocity_commands),
        mean_abs_omega_cmd_rad_s=_mean([abs(value) for value in angular_commands]),
        mean_action_change=_mean(action_changes),
        steering_oscillations=oscillations,
        **totals,
        termination_reason=_optional_text(last_info["termination_reason"]),
        truncation_reason=_optional_text(last_info["truncation_reason"]),
    )


def summarize_lane_episodes(
    rows: Sequence[LaneEpisodeEvaluation],
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty lane evaluation")
    return {
        "episodes": len(rows),
        "lap_success_rate": _rate(row.lap_completed for row in rows),
        "invalid_pose_rate": _rate(row.invalid_pose for row in rows),
        "yellow_crossing_rate": _rate(row.yellow_crossing for row in rows),
        "lane_departure_rate": _rate(row.lane_departure for row in rows),
        "timeout_rate": _rate(row.timeout for row in rows),
        "mean_return": _mean([row.total_return for row in rows]),
        "mean_path_length_m": _mean([row.path_length_m for row in rows]),
        "mean_abs_lateral_error_m": _mean(
            [row.mean_abs_lateral_error_m for row in rows]
        ),
        "mean_p95_abs_lateral_error_m": _mean(
            [row.p95_abs_lateral_error_m for row in rows]
        ),
        "mean_abs_heading_error_rad": _mean(
            [row.mean_abs_heading_error_rad for row in rows]
        ),
        "minimum_yellow_clearance_m": min(
            row.minimum_yellow_clearance_m for row in rows
        ),
        "mean_actual_velocity_mps": _mean(
            [row.mean_actual_velocity_mps for row in rows]
        ),
        "mean_v_cmd_mps": _mean([row.mean_v_cmd_mps for row in rows]),
        "mean_abs_omega_cmd_rad_s": _mean(
            [row.mean_abs_omega_cmd_rad_s for row in rows]
        ),
        "mean_action_change": _mean([row.mean_action_change for row in rows]),
    }


def select_lane_checkpoint(
    scores: Sequence[LaneCheckpointScore],
    *,
    maximum_invalid_pose_rate: float,
    maximum_yellow_crossing_rate: float,
    maximum_lane_departure_rate: float,
) -> LaneCheckpointSelection:
    if not scores:
        raise ValueError("no lane checkpoints to select")
    ordered = sorted(scores, key=lambda item: item.global_step)
    safe = [
        item
        for item in ordered
        if float(item.summary["invalid_pose_rate"]) <= maximum_invalid_pose_rate
        and float(item.summary["yellow_crossing_rate"])
        <= maximum_yellow_crossing_rate
        and float(item.summary["lane_departure_rate"])
        <= maximum_lane_departure_rate
    ]

    def task_key(item: LaneCheckpointScore) -> tuple[float, float, float]:
        return (
            float(item.summary["lap_success_rate"]),
            -float(item.summary["mean_abs_lateral_error_m"]),
            float(item.summary["mean_return"]),
        )

    best_return = max(scores, key=lambda item: float(item.summary["mean_return"]))
    selected = max(safe, key=task_key) if safe else max(scores, key=task_key)
    return LaneCheckpointSelection(
        selected=selected,
        best_return=best_return,
        last=ordered[-1],
        safety_filter_passed=bool(safe),
        reason=(
            "safe candidates ranked by lap success, lane error, then return"
            if safe
            else "no checkpoint passed the predeclared safety filter"
        ),
    )


def lane_acceptance_checks(
    sac: dict[str, object],
    random: dict[str, object],
    always_stop: dict[str, object],
    protocol: LaneProtocol,
) -> dict[str, bool]:
    criteria = protocol.raw["acceptance"]
    sac_success = float(sac["lap_success_rate"])
    return {
        "lap_success": sac_success >= float(criteria["minimum_lap_success_rate"]),
        "no_invalid_pose": float(sac["invalid_pose_rate"])
        <= float(criteria["maximum_invalid_pose_rate"]),
        "no_yellow_crossing": float(sac["yellow_crossing_rate"])
        <= float(criteria["maximum_yellow_crossing_rate"]),
        "no_lane_departure": float(sac["lane_departure_rate"])
        <= float(criteria["maximum_lane_departure_rate"]),
        "mean_lateral_error": float(sac["mean_abs_lateral_error_m"])
        <= float(criteria["maximum_mean_absolute_lateral_error_m"]),
        "p95_lateral_error": float(sac["mean_p95_abs_lateral_error_m"])
        <= float(criteria["maximum_p95_absolute_lateral_error_m"]),
        "mean_heading_error": float(sac["mean_abs_heading_error_rad"])
        <= float(criteria["maximum_mean_absolute_heading_error_rad"]),
        "forward_motion": float(sac["mean_actual_velocity_mps"])
        >= float(criteria["minimum_mean_actual_velocity_mps"]),
        "beats_random": sac_success - float(random["lap_success_rate"])
        >= float(criteria["minimum_lap_success_gain_over_random"]),
        "beats_always_stop": sac_success - float(always_stop["lap_success_rate"])
        >= float(criteria["minimum_lap_success_gain_over_always_stop"]),
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _rate(values) -> float:
    materialized = tuple(bool(value) for value in values)
    return sum(materialized) / len(materialized)


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)

