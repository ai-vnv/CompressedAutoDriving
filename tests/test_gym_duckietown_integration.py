from __future__ import annotations

from dataclasses import fields
from math import isfinite

import numpy as np
import pytest

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    GymDuckietownIntegration,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.coordinates import DUCKIETOWN_COORDINATES
from duckie_pomdp.domain.observation import SensorObservation
from duckie_pomdp.domain.privileged import PrivilegedSimulatorState


@pytest.fixture(scope="module")
def simulator() -> GymDuckietownIntegration:
    pytest.importorskip("gym_duckietown")
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            map_name="small_loop",
            seed=73,
            domain_randomization=False,
            dynamics_randomization=False,
            headless=True,
            start_tile=(1, 0),
            start_pose=((0.065, 0.0, 0.4095), 0.0),
        )
    )
    yield integration
    integration.close()


def test_real_reset_produces_valid_agent_observation(
    simulator: GymDuckietownIntegration,
) -> None:
    observation = simulator.agent.reset(seed=73)

    assert isinstance(observation, SensorObservation)
    assert observation.front_rgb.dtype == np.uint8
    assert observation.front_rgb.ndim == 3
    assert observation.front_rgb.shape[2] == 3
    assert observation.front_rgb.size > 0
    assert all(
        isfinite(value)
        for value in (
            observation.ego.lateral_error_m,
            observation.ego.heading_error_rad,
            observation.ego.linear_velocity_mps,
            observation.ego.yaw_rate_rad_s,
        )
    )


def test_privileged_truth_uses_a_separate_interface(
    simulator: GymDuckietownIntegration,
) -> None:
    observation = simulator.agent.reset(seed=73)
    privileged = simulator.privileged.read()
    agent_fields = {field.name for field in fields(observation)}

    assert isinstance(privileged, PrivilegedSimulatorState)
    assert "true_pomdp_state" not in agent_fields
    assert "ego_world_pose" not in agent_fields
    assert "pedestrian_world_position" not in agent_fields
    assert "pedestrian_world_footprint" not in agent_fields
    assert "stop_sign_world_position" not in agent_fields
    assert "stop_sign_world_footprint" not in agent_fields
    assert privileged.pedestrian_world_position is None
    assert privileged.stop_sign_world_position is None
    assert privileged.stop_line_world_position is None


def test_straight_command_produces_forward_actual_motion(
    simulator: GymDuckietownIntegration,
) -> None:
    simulator.agent.reset(seed=73)
    speeds: list[float] = []
    yaw_rates: list[float] = []
    for _ in range(20):
        transition = simulator.agent.step(PolicyAction(0.2, 0.0))
        speeds.append(transition.observation.ego.linear_velocity_mps)
        yaw_rates.append(transition.observation.ego.yaw_rate_rad_s)

    assert float(np.mean(speeds[-10:])) > 0.0
    assert abs(float(np.mean(yaw_rates[-10:]))) < 0.05


@pytest.mark.parametrize(
    ("omega_command", "expected_sign"),
    [(0.8, 1.0), (-0.8, -1.0)],
)
def test_turn_command_matches_documented_yaw_sign(
    simulator: GymDuckietownIntegration,
    omega_command: float,
    expected_sign: float,
) -> None:
    simulator.agent.reset(seed=73)
    yaw_rates: list[float] = []
    for _ in range(20):
        transition = simulator.agent.step(PolicyAction(0.12, omega_command))
        yaw_rates.append(transition.observation.ego.yaw_rate_rad_s)

    assert expected_sign * float(np.mean(yaw_rates[-10:])) > 0.1


def test_actual_motion_is_measured_separately_from_command(
    simulator: GymDuckietownIntegration,
) -> None:
    simulator.agent.reset(seed=73)
    command = PolicyAction(0.4, 1.0)
    transition = simulator.agent.step(command)
    diagnostics = simulator.diagnostics.read()

    assert diagnostics.requested_action == command
    assert diagnostics.actual_motion == transition.observation.ego.motion
    assert transition.observation.ego.linear_velocity_mps != pytest.approx(
        command.linear_velocity_mps
    )
    assert transition.observation.ego.yaw_rate_rad_s != pytest.approx(
        command.angular_velocity_rad_s
    )


def test_coordinate_conventions_are_single_unit_safe_source() -> None:
    convention = DUCKIETOWN_COORDINATES

    assert convention.distance_unit == "meter"
    assert convention.angle_unit == "radian"
    assert convention.linear_velocity_unit == "meter_per_second"
    assert convention.angular_velocity_unit == "radian_per_second"
    assert convention.ego_lateral_direction == "positive_x_to_left"
    assert convention.positive_yaw_direction == "counter_clockwise"
    assert convention.positive_heading_error_direction == "left_of_lane_tangent"
    assert convention.positive_bearing_direction == "left_of_ego_heading"
