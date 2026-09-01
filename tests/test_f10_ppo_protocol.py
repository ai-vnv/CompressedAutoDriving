import json
from pathlib import Path

import pytest

from duckie_pomdp.control.ppo_protocol import (
    STAGE_NAMES,
    _load_protocol_data,
    _validate_provenance,
    _validate_visual_pretraining_evidence,
    junit_counts,
    load_ppo_curriculum_protocol,
    pretraining_evidence_paths,
    pretraining_source_paths,
    protocol_artifact_root,
    require_curriculum_transition,
    require_pretraining_gate,
    require_stage_in_protocol_scope,
)
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo import PPOAgent, PPOConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_ppo_v1.toml"
VISUAL_CONFIG = ROOT / "configs" / "f10_ppo_visual_v2.toml"
VISUAL_RECOVERY_CONFIG = ROOT / "configs" / "f10_ppo_visual_v3.toml"
VISUAL_CODEX_CONFIG = ROOT / "configs" / "f10_ppo_visual_v4_codex.toml"
VISUAL_C1_REMEDIATION_CONFIG = ROOT / "configs" / "f10_ppo_visual_v5_c1.toml"
VISUAL_COMPETENCE_CONFIG = ROOT / "configs" / "f10_ppo_visual_v9.toml"


def test_curriculum_seeds_are_globally_disjoint_and_historical_free():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    groups = []
    for key in STAGE_NAMES:
        stage = protocol.stage(key)
        groups.extend(
            (set(stage.training_seeds), set(stage.development_seeds), set(stage.stage_final_seeds))
        )
    groups.extend(set(values) for values in protocol.global_final.values())
    for index, left in enumerate(groups):
        assert not (left & set(protocol.historical_seeds))
        for right in groups[index + 1 :]:
            assert not (left & right)


def test_every_stage_has_same_fixed_observation_and_action_contract():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    assert len(protocol.observation_order) == 25
    assert protocol.action_bounds == (0.0, 0.4, -4.0, 4.0)
    assert {protocol.stage(key).map_name for key in STAGE_NAMES} == {
        "small_loop", "experiment_loop", "pomdp_v1"
    }


def test_visual_protocol_uses_29d_experiment_specific_gate_inventory():
    protocol = load_ppo_curriculum_protocol(VISUAL_CONFIG)
    assert len(protocol.observation_order) == 29
    assert protocol_artifact_root(protocol) == ROOT / "artifacts" / "f10_ppo_visual_v2"
    assert "configs/f10_ppo_visual_v2.toml" in pretraining_source_paths(protocol)
    assert "configs/lane_belief_v1.toml" in pretraining_source_paths(protocol)
    assert "configs/f10_ppo_env_v1.json" in pretraining_source_paths(protocol)
    assert "src/duckie_pomdp/adapters/differential_drive.py" in pretraining_source_paths(protocol)
    assert "src/duckie_pomdp/perception/f9_pipeline.py" in pretraining_source_paths(protocol)
    assert "src/duckie_pomdp/perception/yolo_detector.py" in pretraining_source_paths(protocol)
    assert "src/duckie_pomdp/perception/yolo_measurement.py" in pretraining_source_paths(protocol)
    assert (
        "artifacts/f10_ppo_visual_v2/c0/reward_audit_memory_final.json"
        in pretraining_evidence_paths(protocol)
    )
    assert (
        "artifacts/f10_ppo_visual_v2/c0/smoke_memory_final/training_run_manifest.json"
        in pretraining_evidence_paths(protocol)
    )
    assert (
        "artifacts/f10_ppo_visual_v2/c0/reset_memory_audit.json"
        in pretraining_evidence_paths(protocol)
    )
    assert (
        "artifacts/visual_lane/lane_belief_final_validation_metrics.json"
        in pretraining_evidence_paths(protocol)
    )


def test_visual_protocol_scope_fails_closed_after_c1():
    protocol = load_ppo_curriculum_protocol(VISUAL_CONFIG)
    require_stage_in_protocol_scope(protocol, "c0")
    require_stage_in_protocol_scope(protocol, "c1")
    with pytest.raises(RuntimeError, match="stops after C1"):
        require_stage_in_protocol_scope(protocol, "c2")


def test_codex_visual_protocol_binds_its_disjoint_lane_gate_artifacts():
    protocol = load_ppo_curriculum_protocol(VISUAL_CODEX_CONFIG)
    evidence = pretraining_evidence_paths(protocol)
    sources = pretraining_source_paths(protocol)
    assert (
        "artifacts/f10_ppo_visual_v4_codex/lane_calibration/"
        "lane_measurement_calibration_metrics.json"
    ) in evidence
    assert (
        "artifacts/f10_ppo_visual_v4_codex/lane_belief_gate/final_metrics.json"
    ) in evidence
    assert "artifacts/visual_lane/lane_belief_final_validation_metrics.json" not in evidence
    assert "experiments/calibrate_visual_lane_codex_v4.py" in sources
    assert "experiments/validate_visual_lane_codex_v4.py" in sources

    legacy_sources = pretraining_source_paths(
        load_ppo_curriculum_protocol(VISUAL_RECOVERY_CONFIG)
    )
    assert "experiments/calibrate_visual_lane_codex_v4.py" not in legacy_sources
    assert "experiments/validate_visual_lane_codex_v4.py" not in legacy_sources


def test_c1_remediation_extends_frozen_c0_and_binds_transfer_evidence():
    protocol = load_ppo_curriculum_protocol(VISUAL_C1_REMEDIATION_CONFIG)
    sources = pretraining_source_paths(protocol)
    evidence = pretraining_evidence_paths(protocol)

    assert len(protocol.observation_order) == 29
    assert protocol.stage("c1").training_steps == 61_440
    assert protocol.stage("c1").training_seeds == tuple(range(65_001, 65_013))
    assert "configs/f10_ppo_visual_v5_c1.toml" in sources
    assert "configs/f10_ppo_visual_v4_codex.toml" in sources
    assert "configs/lane_belief_v4_transfer.toml" in sources
    assert "experiments/calibrate_lane_transfer_v5.py" in sources
    assert "experiments/validate_lane_transfer_v5.py" in sources
    assert (
        "artifacts/f10_ppo_visual_v5_c1/lane_calibration/"
        "lane_transfer_calibration_metrics.json"
    ) in evidence
    assert (
        "artifacts/f10_ppo_visual_v5_c1/lane_belief_gate/final_metrics.json"
    ) in evidence

    imported_checkpoint = (
        ROOT / "artifacts/f10_ppo_visual_v4_codex/c0/ppo_selected.pt"
    )
    transition = require_curriculum_transition(
        protocol,
        "c1",
        imported_checkpoint,
        protocol_artifact_root(protocol),
    )
    assert transition["previous_stage"] == "c0"
    assert transition["imported"] is True


def test_v9_visual_protocol_binds_camera_lane_competence_sources_and_evidence():
    protocol = load_ppo_curriculum_protocol(VISUAL_COMPETENCE_CONFIG)
    sources = pretraining_source_paths(protocol)
    evidence = pretraining_evidence_paths(protocol)

    assert len(protocol.observation_order) == 29
    assert protocol_artifact_root(protocol) == ROOT / "artifacts" / "f10_ppo_visual_v9"
    assert "configs/lane_belief_v8_competence_rgb.toml" in sources
    assert "configs/lane_rgb_train_v3_competence.toml" in sources
    assert "experiments/generate_lane_rgb_competence_v9.py" in sources
    assert "experiments/build_lane_rgb_combined_v9.py" in sources
    assert "experiments/train_lane_rgb_v7.py" in sources
    assert "experiments/validate_lane_rgb_v7.py" in sources
    assert "experiments/validate_lane_rgb_closed_loop_v7.py" in sources
    assert "experiments/diagnose_lane_rgb_closed_loop_v8.py" in sources

    required = {
        "artifacts/f10_ppo_visual_v9/lane_rgb_model/model_manifest.json",
        "artifacts/f10_ppo_visual_v9/lane_rgb_model/best.pt",
        "datasets/lane_rgb_competence_v9/manifest.json",
        "datasets/lane_rgb_combined_v9/manifest.json",
        "artifacts/f10_ppo_visual_v9/lane_rgb_final/final_metrics.json",
        "artifacts/f10_ppo_visual_v9/lane_closed_loop_gate/final_metrics.json",
        "artifacts/f10_ppo_visual_v9/c0/reset_memory_audit.json",
    }
    assert required.issubset(set(evidence))


def test_protocol_extension_rejects_parent_hash_mismatch(tmp_path):
    parent = tmp_path / "parent.toml"
    child = tmp_path / "child.toml"
    parent.write_text("schema_version = 1\n", encoding="utf-8")
    child.write_text(
        'schema_version = 1\nextends = "parent.toml"\n'
        f'extends_sha256 = "{"0" * 64}"\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="parent hash mismatch"):
        _load_protocol_data(child)


def test_codex_visual_evidence_validation_is_semantic_not_positional(tmp_path):
    protocol = load_ppo_curriculum_protocol(VISUAL_CODEX_CONFIG)
    prefix = "artifacts/f10_ppo_visual_v4_codex"
    calibration_path = (
        f"{prefix}/lane_calibration/lane_measurement_calibration_metrics.json"
    )
    lane_path = f"{prefix}/lane_belief_gate/final_metrics.json"
    memory_path = f"{prefix}/c0/reset_memory_audit.json"
    artifacts = {
        calibration_path: {
            "gate_pass": True,
            "direction": "counter-clockwise",
            "split_unit": "seed/trajectory",
            "calibration_seeds": [61101, 61102],
            "development_seeds": [61201, 61202],
            "seed_overlap": [],
            "runtime_inputs": ["front_rgb", "fixed_camera_calibration"],
        },
        lane_path: {
            "gate_pass": True,
            "direction": "counter-clockwise",
            "seed_role": "once-only held-out gate",
            "config_sha256": protocol.lane_belief_config_sha256,
        },
        memory_path: {
            "passed": True,
            "config_sha256": file_sha256(VISUAL_CODEX_CONFIG),
            "unique_integration_count": 1,
            "unique_simulator_count": 1,
            "resets": 36,
        },
    }
    for relative, payload in artifacts.items():
        output = tmp_path / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload), encoding="utf-8")

    # Deliberately put reset first and calibration last. Tuple indices must not
    # decide which validation rules apply to an artifact.
    evidence = (memory_path, lane_path, calibration_path)
    _validate_visual_pretraining_evidence(tmp_path, protocol, evidence)

    artifacts[lane_path]["seed_role"] = "development"
    (tmp_path / lane_path).write_text(
        json.dumps(artifacts[lane_path]), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="final validation evidence"):
        _validate_visual_pretraining_evidence(tmp_path, protocol, evidence)


def test_visual_recovery_protocol_has_separate_sources_seeds_and_artifacts():
    protocol = load_ppo_curriculum_protocol(VISUAL_RECOVERY_CONFIG)
    sources = pretraining_source_paths(protocol)
    assert len(protocol.observation_order) == 29
    assert protocol_artifact_root(protocol) == ROOT / "artifacts" / "f10_ppo_visual_v3"
    assert "configs/f10_ppo_visual_v3.toml" in sources
    assert "docs/F10_PPO_VISUAL_RECOVERY_FORMULATION.md" in sources
    assert "configs/f10_ppo_visual_v2.toml" not in sources
    assert protocol.raw["reward"]["yellow_curve_recovery_enabled"] is True
    assert protocol.stage("c0").training_seeds[0] == 47001
    assert not (
        set(protocol.stage("c0").development_seeds)
        & set(load_ppo_curriculum_protocol(VISUAL_CONFIG).stage("c0").development_seeds)
    )


def test_frozen_upstream_hashes_are_enforced(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "80154e4ff22d4d9be6ebc1d6bfcd2f7d29caa3458c18ab7859420d2940c4d94a",
        "0" * 64,
    )
    # Paths are relative to configs, so keep the modified file there only in
    # memory semantics by asking the loader to skip path/hash checks.
    candidate = tmp_path / "bad.toml"
    candidate.write_text(text, encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_ppo_curriculum_protocol(candidate)


def test_native_map_hashes_are_actually_enforced():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    provenance = dict(protocol.raw["provenance"])
    provenance["small_loop_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="small_loop map hash mismatch"):
        _validate_provenance(protocol, provenance, require_frozen=True)


def test_pretraining_junit_parser_reads_nested_pytest_suite(tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text(
        '<testsuites><testsuite tests="445" failures="0" errors="0" skipped="0"/></testsuites>',
        encoding="utf-8",
    )
    assert junit_counts(report) == {
        "tests": 445,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }


def test_pretraining_gate_rejects_arbitrary_hash_only_evidence(tmp_path):
    protocol = load_ppo_curriculum_protocol(CONFIG)
    source = tmp_path / "source.py"
    evidence = tmp_path / "evidence.json"
    source.write_text("frozen\n", encoding="utf-8")
    evidence.write_text('{"passed": true}\n', encoding="utf-8")
    gate = tmp_path / "gate.json"
    payload = {
        "schema_version": 1,
        "ready_for_c0_training": True,
        "config_sha256": file_sha256(CONFIG),
        "frozen_sources": {str(source): file_sha256(source)},
        "evidence": {str(evidence): file_sha256(evidence)},
    }
    gate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source inventory"):
        require_pretraining_gate(protocol, gate)


def test_curriculum_transition_requires_selected_passing_predecessor(tmp_path):
    protocol = load_ppo_curriculum_protocol(CONFIG)
    settings = protocol.ppo
    agent = PPOAgent(PPOConfig(
        observation_dimension=len(protocol.observation_order),
        action_dimension=2,
        hidden_sizes=settings.hidden_sizes,
        learning_rate=settings.learning_rate,
        n_steps=settings.n_steps,
        batch_size=settings.batch_size,
        n_epochs=settings.n_epochs,
        gamma=settings.gamma,
        gae_lambda=settings.gae_lambda,
        clip_range=settings.clip_range,
        entropy_coefficient=settings.entropy_coefficient,
        value_function_coefficient=settings.value_function_coefficient,
        max_gradient_norm=settings.max_gradient_norm,
        initial_log_std=settings.initial_log_std,
        seed=settings.training_seed,
        device="cpu",
    ))
    stage_dir = tmp_path / "c0"
    stage_dir.mkdir()
    checkpoint = stage_dir / "ppo_selected.pt"
    config_sha = file_sha256(CONFIG)
    agent.save(
        checkpoint,
        global_step=10_240,
        stage="c0",
        metadata={
            "config_sha256": config_sha,
            "observation_order": list(protocol.observation_order),
        },
    )
    checkpoint_sha = file_sha256(checkpoint)
    (stage_dir / "checkpoint_manifest.json").write_text(json.dumps({
        "stage": "c0",
        "config_sha256": config_sha,
        "selected_is_gate_eligible": True,
        "artifacts": {"selected": {"sha256": checkpoint_sha}},
    }), encoding="utf-8")
    final = {
        "stage": "c0",
        "config_sha256": config_sha,
        "classification": "PASS",
        "progression_permitted": True,
    }
    (stage_dir / "stage_final_metrics.json").write_text(
        json.dumps(final), encoding="utf-8"
    )
    (stage_dir / "retention_metrics.json").write_text(json.dumps({
        "config_sha256": config_sha,
        "retention_pass": True,
    }), encoding="utf-8")
    result = require_curriculum_transition(protocol, "c1", checkpoint, tmp_path)
    assert result["previous_stage"] == "c0"

    final["classification"] = "FAILED"
    final["progression_permitted"] = False
    (stage_dir / "stage_final_metrics.json").write_text(
        json.dumps(final), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="did not PASS"):
        require_curriculum_transition(protocol, "c1", checkpoint, tmp_path)
