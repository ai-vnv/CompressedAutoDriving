from pathlib import Path

from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    pretraining_evidence_paths,
    pretraining_source_paths,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_ppo_visual_objects_v22.toml"


def test_v22_keeps_fixed_belief_contract_and_disjoint_seeds() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    assert len(protocol.observation_order) == 29
    assert protocol.raw["protocol_scope"]["maximum_stage"] == "c4"
    assert protocol.raw["runtime_detection"] == {
        "duckie_minimum_confidence": 0.40,
        "duckie_maximum_bottom_y_px": 240.0,
    }
    c4 = protocol.stage("c4")
    assert not (set(c4.training_seeds) & set(c4.development_seeds))
    assert not (set(c4.training_seeds) & set(c4.stage_final_seeds))
    assert not (set(c4.development_seeds) & set(c4.stage_final_seeds))
    stage_seeds = set(
        c4.training_seeds + c4.development_seeds + c4.stage_final_seeds
    )
    assert not (stage_seeds & set(protocol.historical_seeds))


def test_v22_freezes_c4_public_belief_dagger_evidence_and_generators() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    warm = protocol.raw["behavior_warm_start"]["c4"]
    assert warm["enabled"] is True
    assert warm["dagger"] is True
    sources = set(pretraining_source_paths(protocol))
    evidence = set(pretraining_evidence_paths(protocol))
    assert "experiments/build_ppo_c4_dagger_warm_start.py" in sources
    assert "experiments/validate_ppo_c4_dagger_distillation.py" in sources
    assert "experiments/validate_c4_duckie_image_domain.py" in sources
    assert any(path.endswith("/c4/dagger_distillation_gate.json") for path in evidence)
    assert any(path.endswith("/duckie_bbox_audit.csv") for path in evidence)
    assert any(path.endswith("/duckie_image_domain_gate.json") for path in evidence)


def test_v22_step_zero_remains_ineligible() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    assert int(protocol.raw["checkpoint_selection"]["minimum_updated_global_step"]["c4"]) == 1024
