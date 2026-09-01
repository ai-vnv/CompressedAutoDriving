from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from duckie_pomdp.explain.compressed_policy_analysis import (
    actor_physical,
    normalized_to_physical,
    require_quantized_linear_graph,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f13_explain_compressed_v1.toml"
ARTIFACTS = ROOT / "artifacts/f13_explain_compressed_v1"


def config() -> dict:
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def test_f13_action_mapping_and_actor_contract() -> None:
    assert np.allclose(normalized_to_physical(np.asarray([[-1.0, -1.0], [1.0, 1.0]])), [[0.0, -4.0], [0.4, 4.0]])
    path = (CONFIG.parent / config()["frozen"]["a7"]["checkpoint"]).resolve()
    actor = torch.jit.load(str(path), map_location="cpu").eval()
    require_quantized_linear_graph(actor)
    output = actor_physical(actor, np.zeros((3, 29), dtype=np.float32))
    assert output.shape == (3, 2)
    assert np.isfinite(output).all()


def test_f13_surrogate_gate_blocks_approximation() -> None:
    gate = json.loads((ARTIFACTS / "integrity/surrogate_equivalence.json").read_text())
    assert gate["classification"] == "BLOCKED"
    assert gate["gradient_attribution_authorized"] is False
    assert gate["approximate_surrogate_created"] is False
    assert gate["plausible_exact_qat_state_candidates"] == []


def test_f13_counterfactual_integrity_and_frozen_result() -> None:
    result = json.loads((ARTIFACTS / "counterfactual/counterfactual_metrics.json").read_text())
    assert result["classification"] == "PARTIALLY PRESERVED"
    assert result["sham"]["pass"] is True
    assert result["sham"]["original_maximum_absolute_effect"] == 0.0
    assert result["sham"]["a7_maximum_absolute_effect"] == 0.0
    assert result["primary_checks"]["pedestrian_absent:pedestrian_relevant:v_cmd_mps"] is True
    assert result["primary_checks"]["stop_absent:stop_required:v_cmd_mps"] is False


def test_f13_paired_stress_has_exact_unique_pairs() -> None:
    cfg = config()
    with (ARTIFACTS / "failure_modes/exploratory/paired_episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(row["policy"], int(row["seed"])) for row in rows}
    expected = {(policy, int(seed)) for policy in ("Original", "A7") for seed in cfg["stress"]["exploratory_seeds"]}
    assert len(rows) == len(expected) == 8
    assert keys == expected
    summary = json.loads((ARTIFACTS / "failure_modes/exploratory/summary.json").read_text())
    assert summary["pairing_verified"] is True
    assert summary["behavioral_classification"] == "PRESERVED"
    assert summary["differential_failure_count"] == 0


def test_f13_confirmatory_seeds_remained_unopened() -> None:
    marker = json.loads((ARTIFACTS / "failure_modes/confirmatory_not_run.json").read_text())
    assert marker == {
        "classification": "NOT_RUN",
        "reason": "no exploratory closed-loop compression-related failure candidate",
        "seeds_opened": False,
    }


def test_f13_seed_blocks_are_disjoint() -> None:
    cfg = config()
    exploratory = set(cfg["stress"]["exploratory_seeds"])
    confirmatory = set(cfg["stress"]["confirmatory_seeds"])
    assert exploratory.isdisjoint(confirmatory)
    assert exploratory.isdisjoint(set(range(177001, 177109)))
    assert confirmatory.isdisjoint(set(range(177001, 177109)))
    assert exploratory.isdisjoint(set(range(178001, 178109)))
    assert confirmatory.isdisjoint(set(range(178001, 178109)))


def test_f13_runtime_policy_adapter_has_no_privileged_argument() -> None:
    source = (ROOT / "experiments/run_f13_failure_mode_probe.py").read_text()
    act_block = source[source.index("    def act(self, observation:"):source.index("\n\n\ndef main")]
    assert "evaluation_gt" not in act_block
    assert "privileged" not in act_block
    assert "world_pose" not in act_block
    assert "def act(self, observation" in act_block


def test_f13_final_classifications_are_separate() -> None:
    result = json.loads((ARTIFACTS / "final/explanation_preservation.json").read_text())
    assert result["behavioral_c4"] == "PRESERVED"
    assert result["semantic_explanation_structure"] == "UNRESOLVED"
    assert result["counterfactual_functional_sensitivity"] == "PARTIALLY PRESERVED"
    assert result["overall_f13"] == "LIMITED"
