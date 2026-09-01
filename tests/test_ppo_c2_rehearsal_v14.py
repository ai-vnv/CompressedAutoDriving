from pathlib import Path

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f10_ppo_visual_objects_v14.toml"


def test_v14_rehearsal_schedule_preserves_balanced_crossings() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    rehearsal = protocol.raw["curriculum_rehearsal"]["c2"]
    period = int(rehearsal["period_episodes"])
    offsets = set(int(value) for value in rehearsal["no_pedestrian_offsets"])
    assert period == 3
    assert offsets == {2}
    hazard_indices = [index for index in range(12) if index % period not in offsets]
    assert sum(index % 2 == 0 for index in hazard_indices) == 4
    assert sum(index % 2 == 1 for index in hazard_indices) == 4


def test_v14_rehearsal_schedule_has_exact_documented_sequence() -> None:
    env = PPOCurriculumEnvironment(CONFIG, stage="c2", split="training")
    try:
        actual = []
        for episode_index in range(6):
            env._integration_config(154001 + episode_index, episode_index, episode_index)
            actual.append(
                None if env._c2_rehearsal_no_pedestrian else env._pedestrian_mode
            )
        assert actual == [
            "cross_left_to_right",
            "cross_right_to_left",
            None,
            "cross_left_to_right",
            "cross_right_to_left",
            None,
        ]
    finally:
        env.close()


def test_rehearsal_is_training_only_and_keeps_fixed_observation() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    assert len(protocol.observation_order) == 29
    assert protocol.stage("c2").pedestrian_active is True
    assert "c2_rehearsal_no_pedestrian" not in protocol.observation_order


def test_c2_training_rehearsal_hides_duckie_and_uses_loop_wide_pose() -> None:
    env = PPOCurriculumEnvironment(CONFIG, stage="c2", split="training")
    try:
        hazard_config, _ = env._integration_config(154001, 0, 0)
        reverse_config, _ = env._integration_config(154002, 1, 1)
        rehearsal_config, rehearsal_scenario = env._integration_config(154003, 2, 2)
        assert hazard_config.scenario_pedestrian_enabled is True
        assert reverse_config.scenario_pedestrian_enabled is True
        assert rehearsal_config.scenario_pedestrian_enabled is False
        assert rehearsal_scenario is not None
        assert env._c2_rehearsal_no_pedestrian is True
        assert env._pedestrian_mode is None
        assert rehearsal_scenario.ego_start_tile != (1, 3)
    finally:
        env.close()


def test_c2_development_never_uses_rehearsal() -> None:
    env = PPOCurriculumEnvironment(CONFIG, stage="c2", split="development")
    try:
        config, _ = env._integration_config(154103, 2, 2)
        assert config.scenario_pedestrian_enabled is True
        assert env._c2_rehearsal_no_pedestrian is False
    finally:
        env.close()


def test_rehearsal_uses_explicit_neutral_pedestrian_belief() -> None:
    env = PPOCurriculumEnvironment(CONFIG, stage="c2", split="training")
    try:
        env.reset()
        integration_identity = id(env._integration)
        env.reset()
        observation, info = env.reset()
        assert info["c2_rehearsal_no_pedestrian"] is True
        assert id(env._integration) == integration_identity
        assert env._integration.privileged.read().pedestrian_world_position is None
        belief = env.current_belief.pedestrian
        assert belief.existence_probability == 0.0
        index = env.protocol.observation_order.index("pedestrian_existence_probability")
        assert observation[index] == 0.0

        _, next_info = env.reset()
        assert next_info["c2_rehearsal_no_pedestrian"] is False
        assert id(env._integration) == integration_identity
        assert env._integration.privileged.read().pedestrian_world_position is not None
    finally:
        env.close()
