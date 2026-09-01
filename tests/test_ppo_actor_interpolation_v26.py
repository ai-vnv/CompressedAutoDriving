from __future__ import annotations

from pathlib import Path

import pytest

from experiments.screen_ppo_actor_interpolation_v26 import _parse_seeds


def test_screen_seed_parser_is_explicit_and_nonempty() -> None:
    assert _parse_seeds("1,2,3") == (1, 2, 3)
    with pytest.raises(ValueError, match="at least one"):
        _parse_seeds("")


def test_interpolation_utility_does_not_import_privileged_state() -> None:
    source = Path("experiments/interpolate_ppo_actor_v26.py").read_text(
        encoding="utf-8"
    )
    assert "PrivilegedSimulatorState" not in source
    assert "evaluation_gt" not in source


def test_head_distillation_never_reads_privileged_teacher_columns() -> None:
    source = Path("experiments/distill_ppo_actor_head_v26.py").read_text(
        encoding="utf-8"
    )
    assert 'data["observations"]' in source
    assert 'data["actions"]' in source
    assert "evaluation_gt" not in source
    assert "teacher_gt" not in source
    assert 'agent.model.actor[-1]' in source


def test_conditional_student_dataset_has_no_privileged_output_key() -> None:
    source = Path("experiments/build_ppo_conditional_rehearsal_v27.py").read_text(
        encoding="utf-8"
    )
    save_block = source.split("np.savez_compressed(", 1)[1].split(")\n", 1)[0]
    assert "teacher_gt" not in save_block
    assert "evaluation_gt" not in save_block


def test_trainer_precomputed_checkpoint_is_hash_and_architecture_guarded() -> None:
    source = Path("experiments/train_f10_ppo.py").read_text(encoding="utf-8")
    assert "precomputed behavior checkpoint hash mismatch" in source
    assert "precomputed behavior checkpoint architecture mismatch" in source
    assert '"student_observation_uses_privileged_truth": False' in source


def test_pretraining_protocol_binds_conditional_distillation_sources() -> None:
    source = Path("src/duckie_pomdp/control/ppo_protocol.py").read_text(
        encoding="utf-8"
    )
    assert "build_ppo_conditional_rehearsal_v27.py" in source
    assert "distill_ppo_conditional_actor_v27.py" in source
    assert 'warm_c4["precomputed_checkpoint_sha256"]' in source


def test_belief_gated_dataset_neutralizes_only_public_pedestrian_slice() -> None:
    source = Path("experiments/build_ppo_belief_gated_rehearsal_v28.py").read_text(
        encoding="utf-8"
    )
    assert "PED_SLICE = slice(10, 19)" in source
    assert "teacher_gt" not in source
    assert "evaluation_gt" not in source


def test_pedestrian_adapter_is_structurally_neutral_preserving() -> None:
    source = Path("experiments/distill_ppo_pedestrian_adapter_v29.py").read_text(
        encoding="utf-8"
    )
    assert "PED_START = 10" in source
    assert "PED_STOP = 19" in source
    assert "base_bias - delta @ neutral" in source
    assert "evaluation_gt" not in source
