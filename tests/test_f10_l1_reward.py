from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import (
    LaneRewardConfig,
    LaneRewardEvaluator,
    load_lane_protocol,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import EgoObservation, SensorObservation
from duckie_pomdp.domain.privileged import WorldPose


ROOT = Path(__file__).resolve().parents[1]
CONFIG = LaneRewardConfig.from_protocol(
    load_lane_protocol(ROOT / "configs" / "f10_l1_lane_v1.toml")
)


def observation(*, d: float = 0.0, phi: float = 0.0, speed: float = 0.0):
    return SensorObservation(
        front_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        ego=EgoObservation(d, phi, speed, 0.0),
    )


def evaluate(
    evaluator: LaneRewardEvaluator,
    pose: WorldPose,
    *,
    d: float = 0.0,
    phi: float = 0.0,
    speed: float = 0.0,
    horizon: bool = False,
    curvature: float | None = None,
):
    return evaluator.evaluate(
        action=PolicyAction(speed, 0.0),
        observation=observation(d=d, phi=phi, speed=speed),
        world_pose=pose,
        simulator_terminated=False,
        simulator_truncated=False,
        simulator_done_code="in-progress",
        horizon_reached=horizon,
        road_curvature_inv_m=curvature,
    )


def test_forward_aligned_motion_beats_standing_still() -> None:
    moving = LaneRewardEvaluator(CONFIG, dt_s=1 / 30)
    stopped = LaneRewardEvaluator(CONFIG, dt_s=1 / 30)
    start = WorldPose(0.0, 0.0, 0.0)
    moving.reset(start)
    stopped.reset(start)
    assert evaluate(moving, WorldPose(0.01, 0.0, 0.0), speed=0.2).reward > evaluate(
        stopped, start
    ).reward
    assert evaluate(stopped, start).reward < 0.0


def test_negative_lateral_error_reduces_yellow_clearance() -> None:
    evaluator = LaneRewardEvaluator(CONFIG, dt_s=1 / 30)
    evaluator.reset(WorldPose(0.0, 0.0, 0.0))
    centered = evaluate(evaluator, WorldPose(0.01, 0.0, 0.0), d=0.0)
    near_yellow = evaluate(evaluator, WorldPose(0.02, 0.0, 0.0), d=-0.03)
    assert centered.yellow_clearance_m == pytest.approx(0.042)
    assert near_yellow.yellow_clearance_m == pytest.approx(0.012)
    assert near_yellow.reward_terms.yellow < centered.reward_terms.yellow


def test_footprint_touching_yellow_is_terminal() -> None:
    evaluator = LaneRewardEvaluator(CONFIG, dt_s=1 / 30)
    evaluator.reset(WorldPose(0.0, 0.0, 0.0))
    outcome = evaluate(evaluator, WorldPose(0.01, 0.0, 0.0), d=-0.042)
    assert outcome.yellow_crossing
    assert outcome.terminated
    assert outcome.termination_reason == "yellow_crossing"
    assert outcome.reward_terms.terminal == CONFIG.yellow_crossing_penalty


def test_lap_requires_path_length_leave_and_return_gate() -> None:
    evaluator = LaneRewardEvaluator(CONFIG, dt_s=1 / 30)
    start = WorldPose(0.0, 0.0, 0.0)
    evaluator.reset(start)
    for pose in (
        WorldPose(1.5, 0.0, 0.0),
        WorldPose(1.5, 1.5, 0.0),
        WorldPose(0.0, 1.5, 0.0),
        WorldPose(0.05, 0.0, 0.0),
    ):
        outcome = evaluate(evaluator, pose, speed=0.2)
    assert outcome.path_length_m > CONFIG.minimum_path_length_m
    assert outcome.lap_completed
    assert outcome.termination_reason == "lap_completed"


def test_horizon_is_truncation() -> None:
    evaluator = LaneRewardEvaluator(CONFIG, dt_s=1 / 30)
    start = WorldPose(0.0, 0.0, 0.0)
    evaluator.reset(start)
    outcome = evaluate(evaluator, start, horizon=True)
    assert outcome.truncated and not outcome.terminated
    assert outcome.truncation_reason == "horizon"


RECOVERY_CONFIG = replace(
    CONFIG,
    yellow_curve_recovery_enabled=True,
    yellow_curve_min_abs_curvature_inv_m=0.75,
    yellow_curve_max_penetration_m=0.035,
    yellow_curve_exit_grace_steps=2,
    yellow_recovery_clear_steps=2,
)


def test_shallow_curve_contact_is_soft_and_starts_recovery() -> None:
    evaluator = LaneRewardEvaluator(RECOVERY_CONFIG, dt_s=1 / 30)
    evaluator.reset(WorldPose(0.0, 0.0, 0.0))
    outcome = evaluate(
        evaluator,
        WorldPose(0.01, 0.0, 0.0),
        d=-0.050,
        curvature=2.0,
    )
    assert outcome.yellow_contact
    assert not outcome.yellow_crossing
    assert not outcome.terminated
    assert outcome.yellow_recovery_started
    assert outcome.yellow_recovery_active
    assert outcome.reward_terms.yellow < 0.0
    assert outcome.reward_terms.terminal == 0.0


def test_curve_contact_must_recover_for_consecutive_clear_frames() -> None:
    evaluator = LaneRewardEvaluator(RECOVERY_CONFIG, dt_s=1 / 30)
    evaluator.reset(WorldPose(0.0, 0.0, 0.0))
    evaluate(evaluator, WorldPose(0.01, 0.0, 0.0), d=-0.050, curvature=2.0)
    first_clear = evaluate(
        evaluator, WorldPose(0.02, 0.0, 0.0), d=0.0, curvature=0.0
    )
    recovered = evaluate(
        evaluator, WorldPose(0.03, 0.0, 0.0), d=0.0, curvature=0.0
    )
    assert first_clear.yellow_recovery_active
    assert not first_clear.yellow_recovered
    assert recovered.yellow_recovered
    assert not recovered.yellow_recovery_active
    assert not recovered.terminated


def test_curve_exit_grace_expires_if_contact_persists() -> None:
    evaluator = LaneRewardEvaluator(RECOVERY_CONFIG, dt_s=1 / 30)
    evaluator.reset(WorldPose(0.0, 0.0, 0.0))
    evaluate(evaluator, WorldPose(0.01, 0.0, 0.0), d=-0.050, curvature=2.0)
    tolerated_one = evaluate(
        evaluator, WorldPose(0.02, 0.0, 0.0), d=-0.050, curvature=0.0
    )
    tolerated_two = evaluate(
        evaluator, WorldPose(0.03, 0.0, 0.0), d=-0.050, curvature=0.0
    )
    failed = evaluate(
        evaluator, WorldPose(0.04, 0.0, 0.0), d=-0.050, curvature=0.0
    )
    assert not tolerated_one.terminated
    assert not tolerated_two.terminated
    assert failed.yellow_crossing
    assert failed.terminated
    assert failed.termination_reason == "yellow_recovery_failed"


def test_deep_curve_crossing_is_terminal() -> None:
    evaluator = LaneRewardEvaluator(RECOVERY_CONFIG, dt_s=1 / 30)
    evaluator.reset(WorldPose(0.0, 0.0, 0.0))
    outcome = evaluate(
        evaluator,
        WorldPose(0.01, 0.0, 0.0),
        d=-0.080,
        curvature=2.0,
    )
    assert outcome.yellow_contact
    assert outcome.yellow_crossing
    assert outcome.terminated
    assert outcome.termination_reason == "yellow_crossing"


def test_straight_contact_without_curve_context_is_terminal() -> None:
    evaluator = LaneRewardEvaluator(RECOVERY_CONFIG, dt_s=1 / 30)
    evaluator.reset(WorldPose(0.0, 0.0, 0.0))
    outcome = evaluate(
        evaluator,
        WorldPose(0.01, 0.0, 0.0),
        d=-0.050,
        curvature=0.0,
    )
    assert outcome.yellow_crossing
    assert outcome.terminated
    assert outcome.termination_reason == "yellow_crossing"
