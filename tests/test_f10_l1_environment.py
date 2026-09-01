from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import LaneCurriculumEnvironment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_l1_lane_v1.toml"


def test_lane_environment_rejects_non_split_seed() -> None:
    environment = LaneCurriculumEnvironment(
        CONFIG, split="training", integration_factory=lambda _: None
    )
    try:
        with pytest.raises(ValueError, match="frozen training split"):
            environment.reset(seed=15001)
    finally:
        environment.close()


def test_real_counterclockwise_lane_environment_step(monkeypatch) -> None:
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    environment = LaneCurriculumEnvironment(CONFIG, split="training")
    try:
        observation, reset_info = environment.reset(seed=13001)
        assert reset_info["scenario"] == "small_loop_counterclockwise"
        assert environment.observation_space.contains(observation)
        assert observation.shape == (6,)
        assert np.all(np.isfinite(observation))
        next_observation, reward, terminated, truncated, info = environment.step(
            np.asarray((0.0, 0.0), dtype=np.float32)
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
            "reward_yellow",
            "reward_comfort",
            "reward_living",
            "reward_terminal",
            "yellow_clearance_m",
            "lap_completed",
            "termination_reason",
            "truncation_reason",
        }
        assert not any(
            fragment in key
            for key in info
            for fragment in ("world_pose", "privileged", "truth")
        )
    finally:
        environment.close()
