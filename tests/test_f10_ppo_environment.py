from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import PPOCurriculumEnvironment


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "f10_ppo_v1.toml"
VISUAL_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "f10_ppo_visual_v2.toml"
)


def test_real_c0_uses_full_neutral_belief_vector(monkeypatch):
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    env = PPOCurriculumEnvironment(CONFIG, stage="c0", split="training")
    try:
        observation, _ = env.reset(seed=30001)
        next_observation, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert observation.shape == next_observation.shape == (25,)
        assert np.all(np.isfinite(next_observation))
        assert np.isfinite(reward)
        assert info["policy"]["pedestrian_existence_probability"] == 0.0
        assert info["policy"]["pedestrian_range_mean_m"] > 0.0
        assert info["policy"]["stop_sign_existence_probability"] == 0.0
        assert info["policy"]["stop_mode_none"] == 1.0
        assert info["reward_pedestrian"] == 0.0
        assert info["reward_stop"] == 0.0
    finally:
        env.close()


def test_real_c4_runs_rgb_yolo_belief_before_reward_truth(monkeypatch):
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    env = PPOCurriculumEnvironment(CONFIG, stage="c4", split="training")
    try:
        observation, _ = env.reset(seed=34001)
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert observation.shape == (25,)
        assert info["perception"]["duckie_detection_count"] >= 0
        assert info["perception"]["stop_sign_detection_count"] >= 0
        assert set(info["policy"]) == set(env.protocol.observation_order)
        assert set(info["evaluation_gt"]) == {
            "lane_lateral_error_m",
            "lane_heading_error_rad",
            "road_curvature_inv_m",
            "pedestrian_range_m",
            "pedestrian_bearing_rad",
            "stop_line_distance_m",
        }
        assert not any(name.startswith("evaluation_gt") for name in env.protocol.observation_order)
    finally:
        env.close()


def test_explicit_seed_fixes_pomdp_mode_and_spawn_independent_of_call_order():
    env = PPOCurriculumEnvironment(CONFIG, stage="c2", split="development")
    try:
        seed = 32102
        scenario_index = env._seeds.index(seed)
        first_config, first = env._integration_config(seed, scenario_index)
        # Simulate arbitrary prior resets; explicit seed mapping remains fixed.
        env._episode_index = 19
        second_config, second = env._integration_config(seed, env._seeds.index(seed))
        assert first.pedestrian.mode is second.pedestrian.mode
        assert first.ego_start_pose_m == second.ego_start_pose_m
        assert first.ego_heading_rad == second.ego_heading_rad
        assert first_config.scenario == second_config.scenario
    finally:
        env.close()


def test_c3_configuration_physically_removes_pedestrian():
    env = PPOCurriculumEnvironment(CONFIG, stage="c3", split="development")
    try:
        seed = 33101
        config, _ = env._integration_config(seed, env._seeds.index(seed))
        assert config.scenario_pedestrian_enabled is False
    finally:
        env.close()


def test_real_c3_has_no_pedestrian_truth_or_policy_hazard(monkeypatch):
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    env = PPOCurriculumEnvironment(CONFIG, stage="c3", split="development")
    try:
        observation, _ = env.reset(seed=33101)
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert observation.shape == (25,)
        assert info["evaluation_gt"]["pedestrian_range_m"] is None
        assert info["perception"]["duckie_detection_count"] == 0
        assert info["policy"]["pedestrian_existence_probability"] == 0.0
    finally:
        env.close()


def test_real_visual_c0_runs_lane_belief_and_yolo_from_rgb(monkeypatch):
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    env = PPOCurriculumEnvironment(VISUAL_CONFIG, stage="c0", split="training")
    try:
        observation, reset_info = env.reset(seed=40001)
        next_observation, reward, _, _, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        belief = env.current_belief
        assert observation.shape == next_observation.shape == (29,)
        assert np.all(np.isfinite(next_observation))
        assert np.isfinite(reward)
        assert belief.lane is not None
        assert info["policy"]["lane_lateral_error_mean_m"] == belief.lane.lateral_error_mean_m
        assert info["policy"]["lane_curvature_mean_inv_m"] == belief.lane.curvature_mean_inv_m
        assert info["policy"]["pedestrian_existence_probability"] == 0.0
        assert info["policy"]["stop_sign_existence_probability"] == 0.0
        assert info["policy"]["stop_line_distance_m"] == 2.0
        assert "duckie_detection_count" in info["perception"]
        assert "stop_sign_detection_count" in info["perception"]
        assert "lane_detected" in info["perception"]
        assert set(reset_info["policy"]) == set(env.protocol.observation_order)
        assert not any(
            name.startswith("evaluation_gt")
            for name in env.protocol.observation_order
        )
    finally:
        env.close()


def test_visual_environment_scope_rejects_stage_after_c1():
    with pytest.raises(RuntimeError, match="stops after C1"):
        PPOCurriculumEnvironment(VISUAL_CONFIG, stage="c2", split="training")


def test_visual_native_resets_reuse_one_simulator_context(monkeypatch):
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    env = PPOCurriculumEnvironment(VISUAL_CONFIG, stage="c0", split="training")
    try:
        env.reset(seed=40001)
        integration_id = id(env._integration)
        simulator_id = id(env._integration.agent._session._simulator)
        first_pose = env._integration.agent._session.config.start_pose
        env.reset(seed=40002)
        assert id(env._integration) == integration_id
        assert id(env._integration.agent._session._simulator) == simulator_id
        assert env._integration.agent._session.config.seed == 40002
        assert env._integration.agent._session.config.start_pose != first_pose
    finally:
        env.close()


def test_visual_native_start_is_counter_clockwise():
    env = PPOCurriculumEnvironment(VISUAL_CONFIG, stage="c0", split="development")
    try:
        config, scenario = env._integration_config(40101, 0)
        assert scenario is None
        assert config.start_tile == (1, 0)
        assert config.start_pose is not None
        assert abs(config.start_pose[1] - np.pi) <= 0.04
    finally:
        env.close()
