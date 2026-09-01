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
from experiments.build_ppo_c3_dagger_warm_start import _equal_mass_weights


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f10_ppo_visual_objects_v18.toml"


def test_v18_dagger_dataset_is_balanced_and_contains_no_gt() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    warm = protocol.raw["behavior_warm_start"]["c3"]
    manifest_path = (CONFIG.parent / warm["manifest"]).resolve()
    dataset_path = (CONFIG.parent / warm["dataset"]).resolve()
    source_path = (CONFIG.parent / warm["source_csv"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert warm["dagger"] is True
    assert manifest["source_role"] == "balanced_c2_retention_c3_teacher_and_dagger_v1"
    assert manifest["uses_evaluation_gt"] is False
    assert manifest["observation_dimension"] == len(protocol.observation_order) == 29
    assert manifest["dagger_rows"] > 0
    assert manifest["dagger_stalled_observation_rows"] > 0
    assert manifest["dagger_teacher_drive_rows"] > 0
    masses = np.asarray(
        [
            manifest["anchor_weight_mass"],
            manifest["teacher_weight_mass"],
            manifest["dagger_weight_mass"],
        ]
    )
    assert float(np.max(masses) - np.min(masses)) < 0.1
    assert file_sha256(dataset_path) == warm["dataset_sha256"]
    assert file_sha256(manifest_path) == warm["manifest_sha256"]
    assert file_sha256(source_path) == warm["source_csv_sha256"]
    with np.load(dataset_path) as data:
        assert data["observations"].shape == (manifest["rows"], 29)
        assert data["actions"].shape == (manifest["rows"], 2)
        assert data["weights"].shape == (manifest["rows"],)
        assert all(np.all(np.isfinite(data[name])) for name in data.files)
    with source_path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert not any(name.startswith("evaluation_gt.") for name in header)
    assert sum(name.startswith("policy_normalized.") for name in header) == 29


def test_v18_gate_binds_dagger_sources_checkpoint_and_evidence() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    sources = set(pretraining_source_paths(protocol))
    evidence = set(pretraining_evidence_paths(protocol))
    assert {
        "experiments/build_ppo_c3_dagger_warm_start.py",
        "experiments/validate_ppo_c3_dagger_distillation.py",
    }.issubset(sources)
    assert {
        "artifacts/f10_ppo_visual_objects_v17/c3/training/checkpoints/ppo_c3_step_0000000.pt",
        "artifacts/f10_ppo_visual_objects_v18/c3/dagger_distillation_gate.json",
    }.issubset(evidence)


def test_equal_mass_weights_preserve_role_balance_and_stop_emphasis() -> None:
    targets = np.asarray(
        [(-0.1, 0.0), (-1.0, 0.0), (-0.6, 0.0)], dtype=np.float32
    )
    anchor, teacher, dagger = _equal_mass_weights(5, targets, 7)
    assert np.isclose(anchor.sum(), teacher.sum(), atol=1.0e-5)
    assert np.isclose(anchor.sum(), dagger.sum(), atol=1.0e-5)
    assert teacher[1] > teacher[2] > teacher[0]
    assert np.all(anchor > 0.0) and np.all(dagger > 0.0)
