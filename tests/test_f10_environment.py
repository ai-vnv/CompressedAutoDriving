import inspect
from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import F10BeliefRuntime, F10GymEnvironment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_sac_v1.toml"


def test_f10_runtime_measurement_update_has_no_privileged_parameter() -> None:
    parameters = set(inspect.signature(F10BeliefRuntime.update).parameters)
    assert parameters == {"self", "observation", "previous_action", "dt_s"}


def test_training_environment_rejects_final_evaluation_seed() -> None:
    class UnusedFactory:
        def create(self, _):  # pragma: no cover
            raise AssertionError("factory must not be reached")

    environment = F10GymEnvironment(
        CONFIG,
        split="training",
        integration_factory=lambda _: None,
        belief_runtime_factory=UnusedFactory(),
    )
    try:
        with pytest.raises(ValueError, match="not part of the frozen training split"):
            environment.reset(seed=12001)
    finally:
        environment.close()


def test_real_f10_rgb_yolo_belief_action_step(monkeypatch) -> None:
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    environment = F10GymEnvironment(CONFIG, split="training")
    try:
        observation, reset_info = environment.reset(seed=10001)
        assert environment.observation_space.contains(observation)
        assert observation.shape == (17,)
        assert np.all(np.isfinite(observation))
        assert "perception" in reset_info

        next_observation, reward, terminated, truncated, info = environment.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        assert environment.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["v_cmd"] == pytest.approx(0.2)
        assert info["omega_cmd"] == pytest.approx(0.0)
        assert set(info) >= {
            "reward_progress",
            "reward_lane",
            "reward_stop",
            "reward_pedestrian",
            "reward_comfort",
            "reward_terminal",
            "termination_reason",
            "truncation_reason",
            "perception",
        }
        flattened_names = set(info) | set(info["perception"])
        forbidden = {"privileged", "truth", "world_pose", "gt_bbox", "gt_range"}
        assert flattened_names.isdisjoint(forbidden)
    finally:
        environment.close()
