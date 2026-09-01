from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import (
    load_retention_reference,
    load_ppo_curriculum_protocol,
    pretraining_evidence_paths,
    pretraining_source_paths,
    protocol_artifact_root,
    require_curriculum_transition,
    require_stage_in_protocol_scope,
)
from duckie_pomdp.control.ppo_reward import PPORewardEvaluator
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.evaluation.f10_ppo_policy import BeliefAwareSimpleController
from duckie_pomdp.control.start_sampler import load_small_loop_tiles
from duckie_pomdp.scenario import (
    PedestrianMode,
    load_scenario,
    validate_route_object_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_ppo_visual_objects_v10.toml"


def test_duckie_path_crosses_counter_clockwise_ego_route() -> None:
    scenario = load_scenario(
        ROOT / "configs" / "scenario_experiment_loop_duckie_v1.toml"
    )
    path = scenario.pedestrian.path_for_mode()
    assert path is not None
    geometry = validate_route_object_geometry(
        load_small_loop_tiles(
            map_name="experiment_loop",
            anchor_tile=(1, 0),
            anchor_heading_rad=3.141592653589793,
        ),
        pedestrian_start_world=path[0],
        pedestrian_end_world=path[1],
        stop_line_world=(2.1645, 1.4625),
        stop_sign_world=(2.3, 1.405),
    )
    assert geometry.pedestrian_crosses_route
    assert geometry.pedestrian_intersection_world == pytest.approx((1.4625, 2.1645))
    assert abs(geometry.pedestrian_endpoint_side_a_m) > 0.30
    assert abs(geometry.pedestrian_endpoint_side_b_m) > 0.30
    assert geometry.forward_route_separation_m > 0.75
    assert geometry.euclidean_object_separation_m > 1.0


def test_crossing_direction_reverses_the_same_physical_path() -> None:
    base = load_scenario(
        ROOT / "configs" / "scenario_experiment_loop_duckie_v1.toml"
    )
    left_to_right = base.with_pedestrian_mode(PedestrianMode.CROSS_LEFT_TO_RIGHT)
    right_to_left = base.with_pedestrian_mode(PedestrianMode.CROSS_RIGHT_TO_LEFT)
    left_path = left_to_right.pedestrian.path_for_mode()
    right_path = right_to_left.pedestrian.path_for_mode()
    assert left_path is not None and right_path is not None
    assert left_path == tuple(reversed(right_path))
    assert left_to_right.pedestrian.start_delay_for_mode() == pytest.approx(0.50)
    assert right_to_left.pedestrian.start_delay_for_mode() == pytest.approx(1.55)


def test_c2_c3_are_separate_experiment_loop_scenarios() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    c2 = protocol.stage("c2")
    c3 = protocol.stage("c3")
    assert c2.map_name == c3.map_name == "experiment_loop"
    assert c2.pedestrian_active and not c2.stop_active
    assert c3.stop_active and not c3.pedestrian_active
    assert c2.scenario_config_path != c3.scenario_config_path
    require_stage_in_protocol_scope(protocol, "c3")
    with pytest.raises(RuntimeError, match="stops after C3"):
        require_stage_in_protocol_scope(protocol, "c4")


def test_object_curriculum_binds_stage_specific_pretraining_evidence() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    evidence = set(pretraining_evidence_paths(protocol))
    sources = set(pretraining_source_paths(protocol))
    prefix = "artifacts/f10_ppo_visual_objects_v10"
    assert {
        f"{prefix}/c2/reward_audit.json",
        f"{prefix}/c3/reward_audit.json",
        f"{prefix}/c2/smoke/training_run_manifest.json",
        f"{prefix}/object_scenario_gate.json",
        f"{prefix}/object_reset_memory_audit.json",
    }.issubset(evidence)
    assert "experiments/validate_experiment_loop_objects_v10.py" in sources
    assert "experiments/audit_f10_ppo_object_reset_memory.py" in sources


def test_c2_transition_and_retention_use_the_hash_pinned_imported_c1() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    imported = protocol.raw["curriculum_import"]["c1"]
    checkpoint = (
        protocol.config_path.parent / imported["selected_checkpoint"]
    ).resolve()
    transition = require_curriculum_transition(
        protocol, "c2", checkpoint, protocol_artifact_root(protocol)
    )
    assert transition["imported"] is True
    assert transition["source_checkpoint_sha256"] == file_sha256(checkpoint)

    summary, provenance = load_retention_reference(
        protocol, "c1", protocol_artifact_root(protocol)
    )
    assert summary["completion_rate"] == pytest.approx(1.0)
    assert provenance["reference_imported"] is True
    assert provenance["reference_protocol_sha256"] == imported[
        "source_protocol_sha256"
    ]


def test_experiment_loop_reward_layers_only_the_active_object_component() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    c2 = PPORewardEvaluator(
        protocol, protocol.stage("c2"), dt_s=0.05, route_heading_rad=None
    )
    c3 = PPORewardEvaluator(
        protocol, protocol.stage("c3"), dt_s=0.05, route_heading_rad=None
    )
    assert c2._lane is not None and c2._hazard is not None
    assert c2._hazard.pedestrian_active and not c2._hazard.stop_active
    assert c3._lane is not None and c3._hazard is not None
    assert c3._hazard.stop_active and not c3._hazard.pedestrian_active


def test_reference_controller_stops_for_corridor_hazard_but_not_side_hazard() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    controller = BeliefAwareSimpleController(protocol)
    scales = dict(
        zip(protocol.observation_order, protocol.observation_scales, strict=True)
    )

    def observation(bearing_rad: float) -> np.ndarray:
        values = np.zeros(len(protocol.observation_order), dtype=np.float32)
        physical = {
            "lane_validity_probability": 1.0,
            "pedestrian_existence_probability": 1.0,
            "pedestrian_range_mean_m": 0.60,
            "pedestrian_bearing_mean_rad": bearing_rad,
            "stop_mode_none": 1.0,
        }
        for name, value in physical.items():
            values[protocol.observation_order.index(name)] = value / scales[name]
        return values

    in_path = controller.act(observation(0.0))
    at_side = controller.act(observation(0.90))
    assert in_path[0] == pytest.approx(-1.0)
    assert at_side[0] > in_path[0]


def test_reference_controller_enters_stop_zone_before_commanding_zero() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    controller = BeliefAwareSimpleController(protocol)
    scales = dict(
        zip(protocol.observation_order, protocol.observation_scales, strict=True)
    )

    def observation(stop_distance_m: float) -> np.ndarray:
        values = np.zeros(len(protocol.observation_order), dtype=np.float32)
        physical = {
            "lane_validity_probability": 1.0,
            "stop_line_distance_m": stop_distance_m,
            "stop_mode_required": 1.0,
        }
        for name, value in physical.items():
            values[protocol.observation_order.index(name)] = value / scales[name]
        return values

    approach = controller.act(observation(0.18))
    stop = controller.act(observation(0.14))
    assert approach[0] > -1.0
    assert stop[0] == pytest.approx(-1.0)


def test_c2_resets_reuse_isolated_scenario_simulator(monkeypatch) -> None:
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    env = PPOCurriculumEnvironment(CONFIG, stage="c2", split="training")
    try:
        _, first = env.reset(seed=130001)
        integration = env._integration
        simulator = integration.agent._session._simulator
        _, second = env.reset(seed=130002)
        privileged = integration.privileged.read()
        assert env._integration is integration
        assert env._integration.agent._session._simulator is simulator
        assert first["pedestrian_mode"] != second["pedestrian_mode"]
        assert privileged.true_pomdp_state.pedestrian.exists
        assert privileged.stop_sign_world_position is None
    finally:
        env.close()
