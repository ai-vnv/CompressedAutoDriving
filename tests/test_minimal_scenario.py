from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.scenario import PedestrianMode, load_scenario


SCENARIO_PATH = Path("configs/scenario_pomdp_v1.toml")


def test_f3_real_minimal_scenario_loads_and_steps() -> None:
    scenario = load_scenario(SCENARIO_PATH)
    assert scenario.pedestrian.mode is PedestrianMode.CROSS_LEFT_TO_RIGHT

    integration = create_gym_duckietown(
        GymDuckietownConfig(
            scenario=scenario,
            camera_width=160,
            camera_height=120,
        )
    )
    try:
        observation = integration.agent.reset(seed=scenario.seed)
        assert observation.road is None
        privileged = integration.privileged.read()
        state = privileged.true_pomdp_state

        assert observation.front_rgb.shape == (120, 160, 3)
        assert observation.front_rgb.dtype == np.uint8
        assert privileged.stop_sign_world_position is not None
        assert privileged.stop_line_world_position is not None
        assert privileged.pedestrian_world_position is not None
        assert privileged.pedestrian_world_velocity is not None
        assert state.stop_sign.exists
        assert state.pedestrian.exists
        assert state.road.stop_line_distance_m is not None
        assert state.road.stop_line_distance_m > 0.0

        agent_fields = {field.name for field in fields(observation)}
        assert "true_pomdp_state" not in agent_fields
        assert "stop_sign_world_position" not in agent_fields
        assert "pedestrian_world_position" not in agent_fields

        before_x = privileged.ego_world_pose.x_m
        for _ in range(140):
            transition = integration.agent.step(PolicyAction(0.2, 0.0))
            assert not transition.terminated
            assert not transition.truncated
        after = integration.privileged.read()
        assert after.ego_world_pose.x_m > before_x
        assert after.true_pomdp_state.road.stop_line_distance_m < 0.0
    finally:
        integration.close()


def test_f3_scenario_supports_all_required_pedestrian_modes() -> None:
    scenario = load_scenario(SCENARIO_PATH)

    for mode in PedestrianMode:
        configured = scenario.with_pedestrian_mode(mode)
        assert configured.pedestrian.mode is mode
        if mode is PedestrianMode.STATIONARY:
            assert configured.pedestrian.speed_mps == 0.0
        else:
            assert configured.pedestrian.speed_mps > 0.0
