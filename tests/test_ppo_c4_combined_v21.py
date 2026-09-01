from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    pretraining_evidence_paths,
    protocol_artifact_root,
    require_curriculum_transition,
    require_stage_in_protocol_scope,
)
from experiments import evaluate_f10_ppo
from experiments.train_f10_ppo import _ppo_config
from duckie_pomdp.scenario import load_scenario
from duckie_pomdp.evaluation.f10_ppo_policy import BeliefAwareSimpleController


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_ppo_visual_objects_v21.toml"


def test_v21_opens_only_c4_with_the_same_29d_belief_contract() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    stage = protocol.stage("c4")

    require_stage_in_protocol_scope(protocol, "c4")
    assert len(protocol.observation_order) == 29
    assert stage.pedestrian_active and stage.stop_active
    assert stage.pedestrian_modes == (
        "cross_left_to_right",
        "cross_right_to_left",
    )
    assert stage.training_steps == 8192
    assert stage.checkpoint_interval_steps == 1024
    assert evaluate_f10_ppo._candidate_steps(protocol, stage) == tuple(
        range(1024, 8193, 1024)
    )
    assert stage.scenario_config_path is not None
    scenario = load_scenario(stage.scenario_config_path)
    assert scenario.pedestrian.repeat_crossing is False
    assert scenario.pedestrian.start_delay_s == pytest.approx(11.35)
    assert scenario.pedestrian.reverse_start_delay_s == pytest.approx(13.20)
    assert protocol.raw["runtime_detection"]["duckie_minimum_confidence"] == 0.40


def test_v21_imports_only_the_selected_passing_v20_c3_checkpoint() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    imported = protocol.raw["curriculum_import"]["c3"]
    checkpoint = (protocol.config_path.parent / imported["selected_checkpoint"]).resolve()

    transition = require_curriculum_transition(
        protocol, "c4", checkpoint, protocol_artifact_root(protocol)
    )

    assert transition["imported"] is True
    assert transition["previous_stage"] == "c3"
    assert transition["source_checkpoint_sha256"] == file_sha256(checkpoint)
    assert transition["source_checkpoint_sha256"] == imported[
        "selected_checkpoint_sha256"
    ]


def test_v21_c4_uses_the_kl_guarded_stage_local_ppo() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    c4 = _ppo_config(protocol, "c4", smoke=False)

    assert c4.learning_rate == 1.0e-5
    assert c4.n_epochs == 2
    assert c4.clip_range == 0.05
    assert c4.entropy_coefficient == 0.0
    assert c4.max_gradient_norm == 0.1
    assert c4.target_kl == 0.01
    assert protocol.raw["curriculum_transition"]["c4"] == {
        "reset_optimizer": True,
        "reset_log_std": -3.0,
    }
    assert protocol.raw["checkpoint_selection"]["minimum_updated_global_step"][
        "c4"
    ] == 1024


def test_v21_pretraining_gate_requires_combined_real_runtime_evidence() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    evidence = set(pretraining_evidence_paths(protocol))
    prefix = "artifacts/f10_ppo_visual_objects_v21"

    assert {
        f"{prefix}/c2/reward_audit.json",
        f"{prefix}/c3/reward_audit.json",
        f"{prefix}/c4/reward_audit.json",
        f"{prefix}/c4/smoke/training_run_manifest.json",
        f"{prefix}/object_scenario_gate.json",
        f"{prefix}/object_reset_memory_audit.json",
        f"{prefix}/duckie_confidence_gate.json",
    }.issubset(evidence)
    assert not any("dagger_distillation_gate" in path for path in evidence)


def test_v21_c4_skill_checks_cannot_pass_on_short_or_stationary_behavior() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    summary = {
        "collision_rate": 0.0,
        "unsafe_episode_rate": 0.0,
        "stop_completion_rate": 1.0,
        "stop_violation_rate": 0.0,
        "restart_rate": 1.0,
        "completion_rate": 0.0,
        "mean_progress_m": 0.55,
        "lane_failure_rate": 0.0,
        "stationary_fraction": 1.0,
    }

    checks = evaluate_f10_ppo._skill_checks(protocol, "c4", summary)

    assert checks["completion"] is False
    assert checks["progress"] is False
    assert checks["not_stationary"] is False
    assert not all(checks.values())


def test_v21_c4_retention_checks_all_previous_skills() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    summaries = {
        "c0": {"completion_rate": 1.0, "lane_failure_rate": 0.0},
        "c1": {"completion_rate": 0.25, "lane_failure_rate": 0.75},
        "c2": {"collision_rate": 0.0, "mean_progress_m": 5.0},
        "c3": {
            "stop_completion_rate": 1.0,
            "stop_violation_rate": 0.0,
            "restart_rate": 1.0,
        },
        "c4": {},
    }

    checks, passed = evaluate_f10_ppo._retention_checks(
        protocol, "c4", ROOT / "artifacts" / "f10_ppo_visual_objects_v21", summaries
    )

    assert passed
    assert all(checks.values())


def test_v21_c4_reset_physically_contains_both_objects_and_balances_direction(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    protocol = load_ppo_curriculum_protocol(CONFIG)
    seeds = protocol.stage("c4").training_seeds
    env = PPOCurriculumEnvironment(CONFIG, stage="c4", split="training")
    try:
        vector_a, info_a = env.reset(seed=seeds[0])
        integration = env._integration
        truth_a = integration.privileged.read()
        vector_b, info_b = env.reset(seed=seeds[1])
        truth_b = integration.privileged.read()

        assert vector_a.shape == vector_b.shape == (29,)
        assert env._integration is integration
        assert info_a["pedestrian_mode"] == "cross_left_to_right"
        assert info_b["pedestrian_mode"] == "cross_right_to_left"
        assert truth_a.pedestrian_world_position is not None
        assert truth_a.stop_sign_world_position is not None
        assert truth_b.pedestrian_world_position is not None
        assert truth_b.stop_sign_world_position is not None
        assert info_a["policy"]["stop_line_distance_m"] > 1.5
        assert info_b["policy"]["stop_line_distance_m"] > 1.5
    finally:
        env.close()


def test_v21_c4_duckie_exits_after_exactly_one_crossing(monkeypatch) -> None:
    monkeypatch.setenv("DUCKIETOWN_HEADLESS", "1")
    protocol = load_ppo_curriculum_protocol(CONFIG)
    seed = protocol.stage("c4").training_seeds[0]
    env = PPOCurriculumEnvironment(CONFIG, stage="c4", split="training")
    try:
        env.reset(seed=seed)
        session = env._integration.agent._session
        pedestrian = session._scenario_pedestrian
        assert pedestrian is not None
        pedestrian.pedestrian_wait_time = 0.0
        pedestrian.pedestrian_active = True
        pedestrian.walk_distance = 0.0

        env.step(np.asarray((-1.0, 0.0), dtype=np.float32))

        assert pedestrian not in session._simulator.objects
        assert env._integration.privileged.read().pedestrian_world_position is None
    finally:
        env.close()


def test_v21_reference_ignores_post_stop_false_duckie_but_not_live_crossing() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    controller = BeliefAwareSimpleController(protocol)
    values = dict.fromkeys(protocol.observation_order, 0.0)
    values.update(
        pedestrian_existence_probability=1.0,
        pedestrian_range_mean_m=0.30,
        pedestrian_bearing_mean_rad=0.0,
        lane_validity_probability=1.0,
    )

    live = np.asarray(
        [values[name] / scale for name, scale in zip(
            protocol.observation_order, protocol.observation_scales, strict=True
        )],
        dtype=np.float32,
    )
    post_stop_values = values | {"stop_mode_satisfied": 1.0}
    post_stop = np.asarray(
        [post_stop_values[name] / scale for name, scale in zip(
            protocol.observation_order, protocol.observation_scales, strict=True
        )],
        dtype=np.float32,
    )

    assert controller.act(live)[0] == pytest.approx(-1.0)
    assert controller.act(post_stop)[0] > -0.5
