from pathlib import Path

import pytest

from duckie_pomdp.control import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_ppo_visual_objects_v12.toml"


@pytest.mark.parametrize(
    ("episode", "delay", "phase"),
    ((0, 0.60, 0), (7, 0.60, 0), (8, 1.00, 1), (15, 1.00, 1), (16, 1.55, 2)),
)
def test_right_to_left_training_timing_curriculum(
    episode: int, delay: float, phase: int
) -> None:
    env = PPOCurriculumEnvironment(CONFIG, stage="c2", split="training")
    try:
        _, scenario = env._integration_config(env.stage.training_seeds[1], 1, episode)
        assert scenario is not None
        assert scenario.pedestrian.speed_mps == pytest.approx(0.20)
        assert scenario.pedestrian.start_delay_for_mode() == pytest.approx(delay)
        assert env._pedestrian_training_phase == phase
    finally:
        env.close()


def test_development_uses_full_right_to_left_timing() -> None:
    env = PPOCurriculumEnvironment(CONFIG, stage="c2", split="development")
    try:
        _, scenario = env._integration_config(env.stage.development_seeds[1], 1, 0)
        assert scenario is not None
        assert scenario.pedestrian.start_delay_for_mode() == pytest.approx(1.55)
        assert env._pedestrian_training_phase is None
    finally:
        env.close()


def test_v12_preserves_frozen_boundary_and_acceptance() -> None:
    v10 = load_ppo_curriculum_protocol(ROOT / "configs/f10_ppo_visual_objects_v10.toml")
    v12 = load_ppo_curriculum_protocol(CONFIG)
    assert v12.raw["acceptance"]["c2"] == v10.raw["acceptance"]["c2"]
    assert v12.detector_checkpoint_sha256 == v10.detector_checkpoint_sha256
    assert v12.belief_config_sha256 == v10.belief_config_sha256
    transition = v12.raw["curriculum_transition"]["c2"]
    assert transition["reset_optimizer"] is True
    assert transition["reset_log_std"] == pytest.approx(-1.20)
    assert v12.raw["reward"]["pedestrian_safe_clearance_bonus"] == pytest.approx(3.0)
