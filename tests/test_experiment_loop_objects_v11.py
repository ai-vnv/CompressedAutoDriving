from pathlib import Path

import pytest

from duckie_pomdp.control import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.control.ppo_reward import PPORewardEvaluator


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_ppo_visual_objects_v11.toml"


def _scenario(split: str, episode_index: int):
    env = PPOCurriculumEnvironment(CONFIG, stage="c2", split=split)
    try:
        _, scenario = env._integration_config(
            env.stage.training_seeds[0] if split == "training" else env.stage.development_seeds[0],
            0,
            episode_index,
        )
        assert scenario is not None
        return scenario, env._pedestrian_training_phase
    finally:
        env.close()


@pytest.mark.parametrize(
    ("episode_index", "expected_speed", "expected_phase"),
    ((0, 0.12, 0), (23, 0.12, 0), (24, 0.16, 1), (47, 0.16, 1), (48, 0.20, 2)),
)
def test_c2_training_crossing_speed_curriculum(
    episode_index: int, expected_speed: float, expected_phase: int
) -> None:
    scenario, phase = _scenario("training", episode_index)
    assert scenario.pedestrian.speed_mps == pytest.approx(expected_speed)
    assert phase == expected_phase
    path = scenario.pedestrian.path_for_mode()
    assert path is not None
    assert path[0] != path[1]


def test_c2_development_always_uses_full_crossing_speed() -> None:
    scenario, phase = _scenario("development", 0)
    assert scenario.pedestrian.speed_mps == pytest.approx(0.20)
    assert phase is None


def test_v11_preserves_v10_acceptance_and_frozen_perception() -> None:
    v10 = load_ppo_curriculum_protocol(
        ROOT / "configs" / "f10_ppo_visual_objects_v10.toml"
    )
    v11 = load_ppo_curriculum_protocol(CONFIG)
    assert v11.raw["acceptance"]["c2"] == v10.raw["acceptance"]["c2"]
    assert v11.detector_checkpoint_sha256 == v10.detector_checkpoint_sha256
    assert v11.belief_config_sha256 == v10.belief_config_sha256
    assert v11.lane_belief_config_sha256 == v10.lane_belief_config_sha256
    assert v11.stage("c2").training_steps == 61_440
    assert v11.raw["reward"]["pedestrian_safety_distance_m"] == pytest.approx(0.65)
    assert v11.raw["reward"]["pedestrian_stationary_proximity_factor"] == 0.0
    evaluator = PPORewardEvaluator(
        v11, v11.stage("c2"), dt_s=0.05, route_heading_rad=None
    )
    assert evaluator._hazard is not None
    assert evaluator._hazard.config.pedestrian_stationary_proximity_factor == 0.0


def test_v11_seed_blocks_are_new_and_disjoint() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    active: list[int] = []
    for stage in protocol.stages.values():
        active.extend(stage.training_seeds)
        active.extend(stage.development_seeds)
        active.extend(stage.stage_final_seeds)
    for seeds in protocol.global_final.values():
        active.extend(seeds)
    assert len(active) == len(set(active))
    assert min(active) >= 134001
    assert set(active).isdisjoint(protocol.historical_seeds)
