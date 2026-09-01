from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import LaneTransferEnvironment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_l2_transfer_v1.toml"


def test_transfer_environment_rejects_seed_outside_split() -> None:
    environment = LaneTransferEnvironment(
        CONFIG, split="training", integration_factory=lambda _: None
    )
    try:
        with pytest.raises(ValueError, match="frozen training split"):
            environment.reset(seed=18001)
    finally:
        environment.close()


def test_real_experiment_loop_transfer_step(monkeypatch) -> None:
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    environment = LaneTransferEnvironment(CONFIG, split="training")
    try:
        observation, reset_info = environment.reset(seed=16001)
        assert reset_info["scenario"] == "experiment_loop_mixed_turns"
        assert environment.observation_space.contains(observation)
        next_observation, reward, terminated, truncated, info = environment.step(
            np.asarray((0.0, 0.0), dtype=np.float32)
        )
        assert environment.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["v_cmd"] == pytest.approx(0.2)
        assert info["omega_cmd"] == pytest.approx(0.0)
        assert not any(
            fragment in key
            for key in info
            for fragment in ("world_pose", "privileged", "truth")
        )
    finally:
        environment.close()

