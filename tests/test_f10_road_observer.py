from dataclasses import replace
from inspect import signature
from pathlib import Path

import pytest

from duckie_pomdp.control import F10RoadObserver
from duckie_pomdp.domain.observation import EgoObservation
from duckie_pomdp.scenario import load_scenario


ROOT = Path(__file__).resolve().parents[1]


def test_road_observer_uses_route_prior_and_actual_motion_only() -> None:
    scenario = load_scenario(ROOT / "configs" / "scenario_pomdp_v1.toml")
    observer = F10RoadObserver(scenario, map_tile_size_m=0.585)
    initial = observer.reset()
    expected = scenario.stop_line.world_x_m - (
        scenario.ego_start_tile[0] * 0.585 + scenario.ego_start_pose_m[0]
    )
    assert initial.stop_line_distance_m == pytest.approx(expected)
    updated = observer.update(EgoObservation(0.0, 0.0, 0.3, 0.0), dt_s=0.1)
    assert updated.stop_line_distance_m == pytest.approx(expected - 0.03)
    assert updated.curvature_inv_m == 0.0


def test_road_observer_update_has_no_privileged_or_world_pose_input() -> None:
    parameters = set(signature(F10RoadObserver.update).parameters)
    assert parameters == {"self", "ego", "dt_s"}


def test_road_observer_reset_uses_configured_spawn_not_simulator_pose() -> None:
    scenario = load_scenario(ROOT / "configs" / "scenario_pomdp_v1.toml")
    shifted = replace(
        scenario,
        ego_start_pose_m=(scenario.ego_start_pose_m[0] + 0.2, 0.0, scenario.ego_start_pose_m[2]),
    )
    nominal = F10RoadObserver(scenario, map_tile_size_m=0.585).reset().stop_line_distance_m
    moved = F10RoadObserver(shifted, map_tile_size_m=0.585).reset().stop_line_distance_m
    assert moved == pytest.approx(nominal - 0.2)


def test_route_distance_prior_overrides_straight_projection() -> None:
    scenario = load_scenario(
        ROOT / "configs" / "scenario_experiment_loop_combined_v1.toml"
    )
    observer = F10RoadObserver(
        scenario,
        map_tile_size_m=0.585,
        initial_stop_line_distance_m=1.9,
    )

    assert observer.reset().stop_line_distance_m == pytest.approx(1.9)
