from pathlib import Path

import numpy as np

from duckie_pomdp.control import PPOAgent
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import (
    _validate_pretraining_evidence,
    load_ppo_curriculum_protocol,
    pretraining_evidence_paths,
    pretraining_source_paths,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_ppo_visual_objects_v30.toml"


def test_v30_freezes_29d_existence_gated_public_belief_contract():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    assert len(protocol.observation_order) == 29
    assert (
        protocol.raw["observation"][
            "pedestrian_kinematics_min_existence_probability"
        ]
        == 0.4
    )
    assert protocol.raw["protocol_scope"]["maximum_stage"] == "c4"


def test_v30_pretraining_inventory_binds_distiller_dataset_and_checkpoint():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    sources = set(pretraining_source_paths(protocol))
    evidence = set(pretraining_evidence_paths(protocol))
    assert "experiments/build_ppo_belief_gated_rehearsal_v28.py" in sources
    assert "experiments/distill_ppo_pedestrian_adapter_v29.py" in sources
    assert "experiments/diagnose_c4_policy_divergence.py" in sources
    assert (
        "artifacts/f10_ppo_visual_objects_v30/c4/behavior/"
        "ppo_existence_gated_step0.pt"
    ) in evidence
    assert (
        "artifacts/f10_ppo_visual_objects_v28/behavior/"
        "belief_gated_rehearsal.npz"
    ) in evidence


def test_v30_step_zero_checkpoint_and_npz_are_runtime_leak_free():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    warm = protocol.raw["behavior_warm_start"]["c4"]
    checkpoint = (CONFIG.parent / warm["precomputed_checkpoint"]).resolve()
    assert file_sha256(checkpoint) == warm["precomputed_checkpoint_sha256"]
    agent, payload = PPOAgent.load(checkpoint, device="cpu")
    assert agent.config.observation_dimension == 29
    assert payload["stage"] == "c4"
    assert int(payload["global_step"]) == 0

    dataset = (CONFIG.parent / warm["conditional_dataset"]).resolve()
    with np.load(dataset) as data:
        assert set(data.files) == {
            "observations",
            "actions",
            "weights",
            "value_targets",
            "value_weights",
        }
        assert data["observations"].shape[1] == 29
        assert not any("gt" in key.lower() or "privileged" in key.lower() for key in data.files)


def test_v30_current_pretraining_evidence_semantics_pass():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    _validate_pretraining_evidence(ROOT, protocol)
