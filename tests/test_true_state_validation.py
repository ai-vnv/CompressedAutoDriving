from __future__ import annotations

from pathlib import Path

import pytest

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    GymDuckietownIntegration,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.scenario import PedestrianMode, load_scenario


SCENARIO_PATH = Path("configs/scenario_pomdp_v1.toml")


@pytest.fixture(scope="module")
def stationary_simulator() -> GymDuckietownIntegration:
    scenario = load_scenario(SCENARIO_PATH).with_pedestrian_mode(
        PedestrianMode.STATIONARY
    )
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            scenario=scenario,
            camera_width=80,
            camera_height=60,
        )
    )
    yield integration
    integration.close()


@pytest.fixture(scope="module")
def crossing_simulator() -> GymDuckietownIntegration:
    scenario = load_scenario(SCENARIO_PATH)
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            scenario=scenario,
            camera_width=80,
            camera_height=60,
        )
    )
    yield integration
    integration.close()


def test_f4_manual_relative_geometry_and_stationary_rates(
    stationary_simulator: GymDuckietownIntegration,
) -> None:
    stationary_simulator.agent.reset(seed=123)
    privileged = stationary_simulator.privileged.read()
    state = privileged.true_pomdp_state

    assert state.ego.lateral_error_m == pytest.approx(0.0, abs=1e-12)
    assert state.ego.heading_error_rad == pytest.approx(0.0, abs=1e-12)
    assert state.ego.linear_velocity_mps == 0.0
    assert state.ego.yaw_rate_rad_s == 0.0
    assert state.road.curvature_inv_m == pytest.approx(0.0, abs=1e-12)
    assert state.road.stop_line_distance_m == pytest.approx(1.170 - 0.650)
    assert state.stop_sign.range_m == pytest.approx(0.7107104280)
    assert state.stop_sign.bearing_rad == pytest.approx(-0.2072607107)
    assert state.pedestrian.range_m == pytest.approx(0.9580040827)
    assert state.pedestrian.bearing_rad == pytest.approx(0.3490028742)

    for _ in range(5):
        stationary_simulator.agent.step(PolicyAction(0.0, 0.0))
    stationary = stationary_simulator.privileged.read().true_pomdp_state
    assert stationary.pedestrian.radial_velocity_mps == pytest.approx(0.0)
    assert stationary.pedestrian.bearing_rate_rad_s == pytest.approx(0.0)


def test_f4_forward_ego_produces_negative_relative_range_rate(
    stationary_simulator: GymDuckietownIntegration,
) -> None:
    stationary_simulator.agent.reset(seed=123)
    initial = stationary_simulator.privileged.read().true_pomdp_state.pedestrian
    for _ in range(25):
        stationary_simulator.agent.step(PolicyAction(0.2, 0.0))
    final = stationary_simulator.privileged.read().true_pomdp_state.pedestrian

    assert final.range_m < initial.range_m
    assert final.radial_velocity_mps < 0.0


def test_f4_ego_yaw_changes_stationary_pedestrian_bearing(
    stationary_simulator: GymDuckietownIntegration,
) -> None:
    stationary_simulator.agent.reset(seed=123)
    initial = stationary_simulator.privileged.read().true_pomdp_state.pedestrian
    for _ in range(20):
        stationary_simulator.agent.step(PolicyAction(0.0, 1.0))
    final_state = stationary_simulator.privileged.read().true_pomdp_state

    assert final_state.ego.yaw_rate_rad_s > 0.0
    assert final_state.pedestrian.bearing_rad < initial.bearing_rad
    assert final_state.pedestrian.bearing_rate_rad_s < 0.0


def test_f4_left_to_right_crossing_changes_bearing_sign(
    crossing_simulator: GymDuckietownIntegration,
) -> None:
    crossing_simulator.agent.reset(seed=123)
    initial = crossing_simulator.privileged.read().true_pomdp_state.pedestrian
    for _ in range(80):
        crossing_simulator.agent.step(PolicyAction(0.0, 0.0))
    final = crossing_simulator.privileged.read().true_pomdp_state.pedestrian

    assert initial.bearing_rad > 0.0
    assert final.bearing_rad < 0.0
    assert final.bearing_rate_rad_s < 0.0


def test_right_to_left_mode_spawns_on_right_and_crosses_toward_left() -> None:
    scenario = load_scenario(SCENARIO_PATH).with_pedestrian_mode(
        PedestrianMode.CROSS_RIGHT_TO_LEFT
    )
    integration = create_gym_duckietown(
        GymDuckietownConfig(scenario=scenario, camera_width=80, camera_height=60)
    )
    try:
        integration.agent.reset(seed=scenario.seed)
        initial = integration.privileged.read().true_pomdp_state.pedestrian
        for _ in range(80):
            transition = integration.agent.step(PolicyAction(0.0, 0.0))
            assert not transition.terminated
            assert not transition.truncated
        final = integration.privileged.read().true_pomdp_state.pedestrian
    finally:
        integration.close()

    assert initial.bearing_rad is not None and initial.bearing_rad < 0.0
    assert final.bearing_rad is not None and final.bearing_rad > 0.0


def test_f4_signed_stop_line_distance_decreases_and_crosses_zero(
    stationary_simulator: GymDuckietownIntegration,
) -> None:
    stationary_simulator.agent.reset(seed=123)
    distances = [
        stationary_simulator.privileged.read().true_pomdp_state.road.stop_line_distance_m
    ]
    for _ in range(90):
        transition = stationary_simulator.agent.step(PolicyAction(0.4, 0.0))
        distance = (
            stationary_simulator.privileged.read()
            .true_pomdp_state.road.stop_line_distance_m
        )
        distances.append(distance)
        if distance < -0.05:
            break
        assert not transition.terminated
        assert not transition.truncated

    assert distances[0] > 0.0
    assert all(current <= previous for previous, current in zip(distances, distances[1:]))
    assert distances[-1] < 0.0
