from __future__ import annotations

import importlib.util
import json
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from importlib.machinery import SourceFileLoader

from duckie_pomdp.control.ppo import PPOAgent
from duckie_pomdp.optimization.actor_compression import (
    ActorSpec,
    DenseBeliefActor,
    actor_parameter_count,
    build_pruned_actor,
    convert_qat,
    extract_original_actor,
    physical_actions,
    prepare_ptq,
    prepare_qat,
    require_real_int8,
    save_quantized_actor,
)
from duckie_pomdp.optimization.compression_metrics import action_fidelity


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f12_belief_ppo_compression_v1.toml"
CHECKPOINT = ROOT / "artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt"
SHA = "02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250"


@pytest.fixture(scope="module")
def original():
    return extract_original_actor(CHECKPOINT, expected_sha256=SHA)[0]


def test_original_actor_replays_frozen_ppo_mean(original: DenseBeliefActor) -> None:
    agent, _ = PPOAgent.load(CHECKPOINT, device="cpu")
    observation = np.random.default_rng(12).normal(size=(17, 29)).astype(np.float32)
    with torch.inference_mode():
        expected = agent.model.actor(torch.as_tensor(observation))
        actual = original(torch.as_tensor(observation))
    assert torch.equal(expected, actual)
    assert original.spec == ActorSpec()


@pytest.mark.parametrize("width", (192, 128, 96, 64))
def test_structured_pruning_produces_smaller_dense_actor_from_original(
    original: DenseBeliefActor, width: int
) -> None:
    first = build_pruned_actor(original, width)
    second = build_pruned_actor(original, width)
    assert first.actor.spec.input_dimension == 29
    assert first.actor.spec.hidden_sizes == (width, width)
    assert first.first_layer_survivors == second.first_layer_survivors
    assert first.second_layer_survivors == second.second_layer_survivors
    assert len(first.first_layer_survivors) == width
    assert len(first.second_layer_survivors) == width
    assert actor_parameter_count(first.actor.spec) < actor_parameter_count(original.spec)
    assert first.actor(torch.zeros((3, 29))).shape == (3, 2)


def test_physical_action_mapping_matches_project_contract() -> None:
    mapped = physical_actions(torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]))
    assert torch.allclose(mapped, torch.tensor([[0.0, -4.0], [0.2, 0.0], [0.4, 4.0]]))


def test_ptq_and_qat_produce_real_int8_linear_kernels(original: DenseBeliefActor, tmp_path: Path) -> None:
    calibration = np.random.default_rng(4).normal(size=(128, 29)).astype(np.float32)
    ptq = prepare_ptq(original, calibration)
    require_real_int8(ptq)
    qat = prepare_qat(original)
    converted = convert_qat(qat)
    require_real_int8(converted)
    target = tmp_path / "actor_int8.pt"
    save_quantized_actor(target, ptq)
    reloaded = torch.jit.load(str(target))
    assert reloaded(torch.zeros((2, 29))).shape == (2, 2)


def test_f12_seed_splits_are_disjoint_and_exclude_f11() -> None:
    with CONFIG.open("rb") as stream:
        seeds = tomllib.load(stream)["seeds"]
    names = [name for name in seeds if name != "forbidden_f11_locked"]
    locked = set(seeds["forbidden_f11_locked"])
    for index, name in enumerate(names):
        current = set(seeds[name])
        assert not current & locked
        for other in names[index + 1 :]:
            assert not current & set(seeds[other])


def test_fidelity_metrics_include_tail_correlation_saturation_and_sign() -> None:
    reference = np.asarray([[0.1, -1.0], [0.2, 0.3], [0.3, 1.0]], dtype=np.float32)
    candidate = reference.copy()
    result = action_fidelity(reference, candidate, omega_deadband=0.2)
    assert result["v_cmd_mps"]["maximum_absolute_error"] == 0.0
    assert result["omega_cmd_rad_s"]["p99_absolute_error"] == 0.0
    assert result["v_cmd_mps"]["pearson"] == pytest.approx(1.0)
    assert result["omega_sign"]["eligible_rows"] == 3
    assert result["omega_sign"]["disagreement_frequency"] == 0.0
    assert "action_bound_saturation_frequency" in result


def test_compression_stack_uses_no_new_optional_quantization_dependency() -> None:
    assert importlib.util.find_spec("torch") is not None
    assert importlib.util.find_spec("torchao") is None


def test_original_checkpoint_is_not_under_f12_artifact_root() -> None:
    f12_root = ROOT / "artifacts/f12_belief_ppo_compression_v1"
    assert f12_root not in CHECKPOINT.parents


def test_final_commands_are_fail_closed_behind_claim() -> None:
    source = (ROOT / "experiments/run_f12_compression.py").read_text(encoding="utf-8")
    assert 'if split == "final":\n        require_final_claim' in source
    assert "final fidelity is restricted to the frozen A0-A7 matrix" in source
    assert "final closed-loop is restricted to the frozen A0-A7 matrix" in source
    assert source.count('registry = {key: registry[key] for key in ("A0", selected_key)}') == 2


def test_pareto_frontier_rejects_dominated_candidate() -> None:
    module = SourceFileLoader("f12_runner", str(ROOT / "experiments/run_f12_compression.py")).load_module()
    def fidelity(mae: float):
        return {"overall": {"v_cmd_mps": {"mae": mae}, "omega_cmd_rad_s": {"mae": mae}}}
    metrics = {
        "A": {"logical_parameter_memory_bytes": 100, "actor_checkpoint_size_bytes": 100, "batch1_latency_us_median": 10},
        "B": {"logical_parameter_memory_bytes": 200, "actor_checkpoint_size_bytes": 200, "batch1_latency_us_median": 20},
    }
    assert module.pareto_frontier(["A", "B"], {"A": fidelity(0.01), "B": fidelity(0.02)}, metrics) == ["A"]


def test_final_scope_is_explicitly_c4_only() -> None:
    result_path = ROOT / "artifacts/f12_belief_ppo_compression_v1/final/final_evaluation.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["f12_classification"] == "PASS"
    assert result["deployment_scope"] == "C4 combined scenario only"
    assert result["deployment_authorized"] is True
    assert result["general_cross_curriculum_deployment_authorized"] is False
    assert result["final_c4_fidelity_pass"] is True
    assert result["final_c4_behavior_pass"] is True


def test_a7_video_renderer_is_qualitative_and_never_uses_final_seeds() -> None:
    source = (ROOT / "experiments/render_f12_compressed_bev.py").read_text(encoding="utf-8")
    assert 'default=178021' in source
    assert 'seed_split": "compression_selection"' in source
    assert '"role": "qualitative_example_only"' in source
    assert 'BEV/GT are visualization-only; actor input is public 29D belief' in source
    assert 'actor(torch.as_tensor(observation' in source
    assert 'qualitative video seed must come from compression-selection, never final holdout' in source
