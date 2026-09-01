import csv
import json
from pathlib import Path

import numpy as np

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    pretraining_evidence_paths,
    pretraining_source_paths,
)
from duckie_pomdp.scenario import load_scenario


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f10_ppo_visual_objects_v17.toml"


def test_v17_c3_warm_start_is_balanced_public_belief_data() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    warm = protocol.raw["behavior_warm_start"]["c3"]
    assert protocol.raw["behavior_warm_start"]["c2"]["enabled"] is False
    assert warm["enabled"] is True
    assert len(protocol.observation_order) == 29

    dataset = (CONFIG.parent / warm["dataset"]).resolve()
    manifest_path = (CONFIG.parent / warm["manifest"]).resolve()
    source_csv = (CONFIG.parent / warm["source_csv"]).resolve()
    assert file_sha256(dataset) == warm["dataset_sha256"]
    assert file_sha256(manifest_path) == warm["manifest_sha256"]
    assert file_sha256(source_csv) == warm["source_csv_sha256"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["uses_evaluation_gt"] is False
    assert manifest["anchor_rows"] == manifest["stop_rows"] > 0
    assert manifest["observation_dimension"] == 29
    assert manifest["stop_action_rows"] > 0
    assert manifest["satisfied_observation_rows"] > 0
    assert manifest["detected_steps"] > 0
    assert abs(manifest["anchor_weight_mass"] - manifest["stop_weight_mass"]) < 0.1
    assert all(
        episode["completed"]
        and episode["stop_completed"]
        and episode["restarted"]
        and not episode["stop_violation"]
        and not episode["collision"]
        and not episode["invalid_pose"]
        for episode in manifest["source_episodes"]
    )
    with np.load(dataset) as data:
        assert data["observations"].shape == (manifest["rows"], 29)
        assert data["actions"].shape == (manifest["rows"], 2)
        assert data["weights"].shape == (manifest["rows"],)
        assert all(np.all(np.isfinite(data[name])) for name in data.files)

    with source_csv.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert not any(name.startswith("evaluation_gt.") for name in header)
    assert sum(name.startswith("policy_normalized.") for name in header) == 29


def test_v17_gate_binds_c3_builder_and_artifacts() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    sources = set(pretraining_source_paths(protocol))
    evidence = set(pretraining_evidence_paths(protocol))
    assert "experiments/build_ppo_c3_behavior_warm_start.py" in sources
    assert {
        "artifacts/f10_ppo_visual_objects_v16/c3/behavior/behavior_warm_start.npz",
        "artifacts/f10_ppo_visual_objects_v16/c3/behavior/behavior_warm_start_manifest.json",
        "artifacts/f10_ppo_visual_objects_v16/c3/behavior/behavior_warm_start_source.csv",
        "artifacts/f10_ppo_visual_objects_v17/c3/smoke/training_run_manifest.json",
    }.issubset(evidence)


def test_v17_reward_audits_bind_their_scenario_and_map() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    for stage_key in ("c2", "c3"):
        stage = protocol.stage(stage_key)
        assert stage.scenario_config_path is not None
        scenario = load_scenario(stage.scenario_config_path)
        artifact = json.loads(
            (
                ROOT
                / "artifacts/f10_ppo_visual_objects_v17"
                / stage_key
                / "reward_audit.json"
            ).read_text(encoding="utf-8")
        )
        assert artifact["passed"] is True
        assert artifact["config_sha256"] == file_sha256(CONFIG)
        assert artifact["scenario_provenance"]["config_sha256"] == file_sha256(
            stage.scenario_config_path
        )
        assert artifact["scenario_provenance"]["map_sha256"] == file_sha256(
            scenario.map_path
        )
    stop_checks = json.loads(
        (
            ROOT
            / "artifacts/f10_ppo_visual_objects_v17/c3/reward_audit.json"
        ).read_text(encoding="utf-8")
    )["checks"]
    assert stop_checks["simple_stops"]
    assert stop_checks["simple_restarts_after_stop"]
    assert stop_checks["simple_does_not_violate_stop"]
    assert stop_checks["simple_no_collision"]
    assert stop_checks["simple_no_invalid_pose"]
