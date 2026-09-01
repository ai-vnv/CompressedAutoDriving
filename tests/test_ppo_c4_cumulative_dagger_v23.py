from pathlib import Path

import numpy as np

from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    pretraining_evidence_paths,
    pretraining_source_paths,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f10_ppo_visual_objects_v23.toml"


def test_v23_keeps_fixed_public_belief_and_fresh_c4_splits() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    assert len(protocol.observation_order) == 29
    assert protocol.raw["behavior_warm_start"]["c4"]["cumulative_policy_rehearsal"] is True
    stage = protocol.stage("c4")
    assert stage.training_seeds == tuple(range(171001, 171013))
    assert stage.development_seeds == tuple(range(171101, 171105))
    assert stage.stage_final_seeds == tuple(range(171201, 171205))


def test_v23_dataset_is_finite_public_29d_with_four_roles() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    warm = protocol.raw["behavior_warm_start"]["c4"]
    dataset = (CONFIG.parent / warm["dataset"]).resolve()
    with np.load(dataset) as data:
        assert data["observations"].shape == (24875, 29)
        assert data["actions"].shape == (24875, 2)
        assert data["weights"].shape == (24875,)
        assert all(np.all(np.isfinite(data[key])) for key in data.files)
        assert np.all(data["weights"] > 0.0)


def test_v23_freezes_cumulative_generators_and_evidence() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    sources = set(pretraining_source_paths(protocol))
    assert "experiments/build_ppo_c4_cumulative_dagger_v23.py" in sources
    assert "experiments/validate_ppo_c4_cumulative_dagger_v23.py" in sources
    assert "experiments/visualize_ppo_behavior_dataset.py" in sources
    assert "experiments/build_ppo_c4_dagger_warm_start.py" not in sources
    evidence = set(pretraining_evidence_paths(protocol))
    assert "artifacts/f10_ppo_visual_objects_v23/c4/behavior/behavior_warm_start.npz" in evidence
    assert "artifacts/f10_ppo_visual_objects_v23/c4/dagger_distillation_gate.json" in evidence
