from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import load_f10_protocol
from duckie_pomdp.evaluation.f10_policy import (
    AlwaysStopPolicy,
    CheckpointScore,
    EpisodeEvaluation,
    SimpleControllerPolicy,
    acceptance_checks,
    run_episode,
    select_checkpoint,
    summarize_episodes,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_sac_v1.toml"


def _score(
    step: int,
    *,
    collision: float,
    invalid: float,
    success: float,
    progress: float,
    episode_return: float,
) -> CheckpointScore:
    return CheckpointScore(
        path=Path(f"step_{step}.pt"),
        global_step=step,
        checkpoint_sha256=str(step),
        summary={
            "collision_rate": collision,
            "invalid_pose_rate": invalid,
            "success_rate": success,
            "mean_progress_m": progress,
            "mean_return": episode_return,
        },
    )


def test_safety_first_selection_rejects_higher_return_unsafe_checkpoint() -> None:
    unsafe = _score(
        4000,
        collision=0.40,
        invalid=0.0,
        success=1.0,
        progress=2.0,
        episode_return=100.0,
    )
    safe = _score(
        8000,
        collision=0.0,
        invalid=0.0,
        success=0.5,
        progress=1.0,
        episode_return=5.0,
    )
    selection = select_checkpoint(
        (unsafe, safe),
        maximum_collision_rate=0.20,
        maximum_invalid_pose_rate=0.25,
    )
    assert selection.selected is safe
    assert selection.best_return is unsafe
    assert selection.last is safe
    assert selection.safety_filter_passed


def test_no_safe_checkpoint_is_diagnostic_and_capped() -> None:
    first = _score(
        4000,
        collision=0.30,
        invalid=0.50,
        success=0.0,
        progress=0.2,
        episode_return=-3.0,
    )
    least_unsafe = _score(
        8000,
        collision=0.25,
        invalid=0.40,
        success=0.1,
        progress=0.4,
        episode_return=-2.0,
    )
    selection = select_checkpoint(
        (first, least_unsafe),
        maximum_collision_rate=0.20,
        maximum_invalid_pose_rate=0.25,
    )
    assert selection.selected is least_unsafe
    assert not selection.safety_filter_passed
    assert "LIMITED" in selection.selection_reason


def test_baseline_policies_use_normalized_action_contract() -> None:
    protocol = load_f10_protocol(CONFIG)
    observation = np.zeros(17, dtype=np.float32)
    stop = AlwaysStopPolicy()
    stop.reset(1)
    assert np.array_equal(stop.act(observation), np.array([-1.0, 0.0], np.float32))

    simple = SimpleControllerPolicy(protocol)
    simple.reset(1)
    action = simple.act(observation)
    assert action.shape == (2,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


class _TwoStepEnvironment:
    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, *, seed: int):
        self.step_count = 0
        return np.zeros(17, np.float32), {"seed": seed, "scenario": "stationary"}

    def step(self, action):
        self.step_count += 1
        done = self.step_count == 2
        info = {
            "reward_progress": 0.1,
            "reward_lane": -0.01,
            "reward_stop": 0.0,
            "reward_pedestrian": 0.0,
            "reward_comfort": 0.0,
            "reward_terminal": 2.0 if done else 0.0,
            "progress_m": 1.3 if done else 0.2,
            "pedestrian_clearance_m": 0.30 if self.step_count == 1 else 0.50,
            "collision": False,
            "unsafe_proximity": self.step_count == 1,
            "lane_departure": False,
            "stop_completed": done,
            "stop_violation": False,
            "invalid_pose": False,
            "termination_reason": "success" if done else None,
            "truncation_reason": None,
            "v_cmd": 0.1 if self.step_count == 1 else 0.3,
            "omega_cmd": 0.0,
        }
        return np.zeros(17, np.float32), 0.09 + (2.0 if done else 0.0), done, False, info


class _ZeroPolicy:
    name = "zero"

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation):
        del observation
        return np.zeros(2, np.float32)


def test_episode_evaluator_uses_public_environment_boundary() -> None:
    protocol = load_f10_protocol(CONFIG)
    row = run_episode(
        _TwoStepEnvironment(),
        seed=11001,
        policy=_ZeroPolicy(),
        protocol=protocol,
        checkpoint_step=4000,
    )
    assert row.success
    assert row.steps == 2
    assert row.safety_region_steps == 1
    assert row.clear_region_steps == 1
    summary = summarize_episodes([row])
    assert summary["success_rate"] == pytest.approx(1.0)
    assert summary["pedestrian_speed_response_mps"] == pytest.approx(0.2)


def test_acceptance_is_predeclared_and_does_not_use_final_data_to_fit() -> None:
    protocol = load_f10_protocol(CONFIG)
    sac = {
        "success_rate": 0.5,
        "mean_progress_m": 1.0,
        "collision_rate": 0.0,
        "lane_departure_episode_rate": 0.1,
        "pedestrian_speed_response_mps": 0.05,
    }
    random = {"mean_progress_m": 0.5, "lane_departure_episode_rate": 0.5}
    stop = {"mean_progress_m": 0.0}
    assert all(acceptance_checks(sac, random, stop, protocol).values())
