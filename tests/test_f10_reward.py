from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import F10RewardConfig, F10RewardEvaluator, load_f10_protocol
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import EgoObservation, RoadMeasurement, SensorObservation
from duckie_pomdp.domain.privileged import (
    PrivilegedSimulatorState,
    WorldFootprint,
    WorldPoint,
    WorldPose,
)
from duckie_pomdp.domain.state import (
    EgoState,
    POMDPState,
    PedestrianState,
    RoadState,
    StopMode,
    StopSignState,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = F10RewardConfig.from_protocol(
    load_f10_protocol(ROOT / "configs" / "f10_sac_v1.toml")
)


def observation(
    x: float,
    speed: float,
    *,
    d: float = 0.0,
    stop_distance: float | None = None,
) -> SensorObservation:
    return SensorObservation(
        front_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        ego=EgoObservation(d, 0.0, speed, 0.0),
        road=RoadMeasurement(
            0.0,
            1.0 - x if stop_distance is None else stop_distance,
        ),
    )


def privileged(x: float, *, pedestrian_x: float | None = None) -> PrivilegedSimulatorState:
    footprint = None
    pedestrian_position = None
    if pedestrian_x is not None:
        pedestrian_position = WorldPoint(pedestrian_x, 0.0)
        footprint = WorldFootprint(
            tuple(
                WorldPoint(pedestrian_x + dx, dz)
                for dx, dz in (
                    (-0.03, -0.03),
                    (0.03, -0.03),
                    (0.03, 0.03),
                    (-0.03, 0.03),
                )
            )
        )
    state = POMDPState(
        ego=EgoState(0.0, 0.0, 0.0, 0.0),
        road=RoadState(0.0, 1.0 - x, StopMode.NONE),
        stop_sign=StopSignState(False, None, None),
        pedestrian=(
            PedestrianState(False, None, None, None, None)
            if pedestrian_x is None
            else PedestrianState(True, abs(pedestrian_x - x), 0.0, 0.0, 0.0)
        ),
    )
    return PrivilegedSimulatorState(
        true_pomdp_state=state,
        ego_world_pose=WorldPose(x, 0.0, 0.0),
        stop_sign_world_position=None,
        stop_sign_world_footprint=None,
        stop_line_world_position=WorldPoint(1.0, 0.0),
        pedestrian_world_position=pedestrian_position,
        pedestrian_world_footprint=footprint,
        pedestrian_world_velocity=None,
        collision=None,
    )


def step(
    evaluator,
    x,
    speed,
    *,
    pedestrian_x=None,
    done=False,
    code="in-progress",
    horizon=False,
):
    return evaluator.evaluate(
        action=PolicyAction(speed, 0.0),
        observation=observation(x, speed),
        privileged=privileged(x, pedestrian_x=pedestrian_x),
        simulator_terminated=done,
        simulator_truncated=False,
        simulator_done_code=code,
        horizon_reached=horizon,
    )


def test_forward_progress_beats_standing_still() -> None:
    moving = F10RewardEvaluator(CONFIG, route_heading_rad=0.0)
    stopped = F10RewardEvaluator(CONFIG, route_heading_rad=0.0)
    for evaluator in (moving, stopped):
        evaluator.reset(observation(0.0, 0.0), privileged(0.0))
    assert step(moving, 0.1, 0.2).reward > step(stopped, 0.0, 0.0).reward
    assert step(stopped, 0.0, 0.0).reward < 0.0


def test_lane_error_is_penalized() -> None:
    evaluator = F10RewardEvaluator(CONFIG, route_heading_rad=0.0)
    evaluator.reset(observation(0.0, 0.0), privileged(0.0))
    outcome = evaluator.evaluate(
        action=PolicyAction(0.0, 0.0),
        observation=observation(0.0, 0.0, d=0.25),
        privileged=privileged(0.0),
        simulator_terminated=False,
        simulator_truncated=False,
        simulator_done_code="in-progress",
        horizon_reached=False,
    )
    assert outcome.reward_terms.lane < 0.0
    assert outcome.lane_departure
    assert outcome.terminated
    assert outcome.termination_reason == "lane_departure"
    assert outcome.reward_terms.terminal == CONFIG.invalid_pose_terminal_penalty


def test_stop_completion_requires_consecutive_low_speed_frames() -> None:
    evaluator = F10RewardEvaluator(CONFIG, route_heading_rad=0.0)
    evaluator.reset(observation(0.75, 0.0, stop_distance=0.25), privileged(0.75))
    outcomes = [step(evaluator, 0.75, 0.0) for _ in range(CONFIG.stop_hold_steps)]
    assert not any(item.stop_completed for item in outcomes[:-1])
    assert outcomes[-1].stop_completed
    assert outcomes[-1].reward_terms.stop >= CONFIG.stop_completion_bonus


def test_crossing_without_stop_is_a_violation() -> None:
    evaluator = F10RewardEvaluator(CONFIG, route_heading_rad=0.0)
    evaluator.reset(observation(0.95, 0.2), privileged(0.95))
    outcome = step(evaluator, 1.01, 0.2)
    assert outcome.stop_violation
    assert outcome.reward_terms.stop <= CONFIG.stop_violation_penalty


def test_pedestrian_contact_is_terminal_and_not_hidden_as_a_miss() -> None:
    evaluator = F10RewardEvaluator(CONFIG, route_heading_rad=0.0)
    evaluator.reset(observation(0.0, 0.0), privileged(0.0, pedestrian_x=0.02))
    outcome = step(evaluator, 0.0, 0.1, pedestrian_x=0.02)
    assert outcome.pedestrian_clearance_m == pytest.approx(0.0)
    assert outcome.pedestrian_collision
    assert outcome.terminated
    assert outcome.termination_reason == "pedestrian_collision"
    assert outcome.reward_terms.terminal == CONFIG.pedestrian_collision_terminal_penalty


def test_horizon_is_truncation_not_termination() -> None:
    evaluator = F10RewardEvaluator(CONFIG, route_heading_rad=0.0)
    evaluator.reset(observation(0.0, 0.0), privileged(0.0))
    outcome = step(evaluator, 0.0, 0.0, horizon=True)
    assert outcome.truncated and not outcome.terminated
    assert outcome.truncation_reason == "horizon"
