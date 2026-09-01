import json
from pathlib import Path

import numpy as np

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    pretraining_evidence_paths,
    pretraining_source_paths,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f10_ppo_visual_objects_v13.toml"


def test_behavior_dataset_contains_only_fixed_policy_vectors_and_actions() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    settings = protocol.raw["behavior_warm_start"]["c2"]
    path = (protocol.config_path.parent / settings["dataset"]).resolve()
    assert file_sha256(path) == settings["dataset_sha256"]
    with np.load(path) as data:
        assert set(data.files) == {"observations", "actions", "weights"}
        assert data["observations"].shape == (25_600, 29)
        assert data["actions"].shape == (25_600, 2)
        assert np.all(np.isfinite(data["observations"]))
        assert np.all((data["actions"] >= -1.0) & (data["actions"] <= 1.0))


def test_behavior_manifest_explicitly_excludes_evaluation_truth() -> None:
    manifest = json.loads((
        ROOT / "artifacts/f10_ppo_visual_objects_v13/c2/behavior_warm_start_manifest.json"
    ).read_text())
    assert manifest["uses_evaluation_gt"] is False
    assert manifest["observation_dimension"] == 29
    assert manifest["stop_action_rows"] == 8_392
    snapshot = Path(manifest["source_csv"])
    assert snapshot == (
        ROOT
        / "artifacts/f10_ppo_visual_objects_v13/c2/behavior_warm_start_source.csv"
    )
    assert file_sha256(snapshot) == manifest["source_csv_sha256"]
    assert manifest["source_snapshot_rows"] == manifest["rows"] == 25_600


def test_behavior_inputs_and_generator_are_frozen_by_pretraining_gate() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    evidence = set(pretraining_evidence_paths(protocol))
    sources = set(pretraining_source_paths(protocol))
    assert "artifacts/f10_ppo_visual_objects_v13/c2/behavior_warm_start.npz" in evidence
    assert "artifacts/f10_ppo_visual_objects_v13/c2/behavior_warm_start_manifest.json" in evidence
    assert "artifacts/f10_ppo_visual_objects_v13/c2/behavior_warm_start_source.csv" in evidence
    assert "experiments/build_ppo_behavior_warm_start.py" in sources
    assert "experiments/freeze_behavior_source_prefix.py" in sources
