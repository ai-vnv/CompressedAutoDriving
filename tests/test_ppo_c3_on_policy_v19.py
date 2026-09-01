import json
from pathlib import Path

import pytest

from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from experiments.train_f10_ppo import _ppo_config
from experiments import evaluate_f10_ppo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f10_ppo_visual_objects_v19.toml"


def test_v19_resets_exploration_and_requires_an_updated_c3_checkpoint() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    stage = protocol.stage("c3")
    transition = protocol.raw["curriculum_transition"]["c3"]

    assert stage.training_steps == 8192
    assert stage.checkpoint_interval_steps == 1024
    assert transition == {"reset_optimizer": True, "reset_log_std": -3.0}
    assert protocol.raw["checkpoint_selection"]["minimum_updated_global_step"]["c3"] == 1024
    assert evaluate_f10_ppo._candidate_steps(protocol, stage) == tuple(
        range(0, 8193, 1024)
    )


def test_stage_final_refuses_a_diagnostic_only_selection(tmp_path, monkeypatch) -> None:
    stage_dir = tmp_path / "c3"
    stage_dir.mkdir()
    (stage_dir / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": "irrelevant-before-hash-check",
                "selected_is_gate_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    (stage_dir / "retention_metrics.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        evaluate_f10_ppo,
        "file_sha256",
        lambda path: "irrelevant-before-hash-check",
    )
    with pytest.raises(RuntimeError, match="eligible development checkpoint"):
        evaluate_f10_ppo.stage_final(CONFIG, "c3", stage_dir, device="cpu")


def test_v20_uses_conservative_stage_local_ppo_without_changing_c2_contract() -> None:
    config = ROOT / "configs/f10_ppo_visual_objects_v20.toml"
    protocol = load_ppo_curriculum_protocol(config)
    c2 = _ppo_config(protocol, "c2", smoke=False)
    c3 = _ppo_config(protocol, "c3", smoke=False)

    assert c2.learning_rate == protocol.ppo.learning_rate == 3.0e-4
    assert c2.target_kl is None
    assert c3.learning_rate == 1.0e-5
    assert c3.n_epochs == 2
    assert c3.clip_range == 0.05
    assert c3.entropy_coefficient == 0.0
    assert c3.max_gradient_norm == 0.1
    assert c3.target_kl == 0.01
