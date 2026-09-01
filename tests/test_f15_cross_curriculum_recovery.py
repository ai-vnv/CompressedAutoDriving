from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.optimization.compression_metrics import action_fidelity

from duckie_pomdp.optimization.cross_curriculum_recovery import (
    CURRICULA,
    absolute_retention_checks,
    curriculum_balanced_probabilities,
    first_objective_failure_event,
    retention_decision,
    validate_seed_partition,
    verify_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def _summary(**updates):
    base = {
        "completion_rate": 1.0,
        "lane_failure_rate": 0.0,
        "invalid_pose_rate": 0.0,
        "mean_abs_lateral_error_m": 0.02,
        "mean_progress_m": 8.0,
        "collision_rate": 0.0,
        "collision_episodes": 0,
        "unsafe_episode_rate": 0.0,
        "stationary_fraction": 0.1,
        "stop_completion_rate": 1.0,
        "stop_violation_rate": 0.0,
        "restart_rate": 1.0,
        "minimum_pedestrian_clearance_m": 0.5,
    }
    base.update(updates)
    return base


def _thresholds():
    return {
        "c0": {"minimum_completion_rate": 0.5},
        "c1": {"minimum_completion_rate": 0.5, "maximum_small_loop_completion_drop": 0.25},
        "c2": {"minimum_mean_progress_m": 4.5, "maximum_c1_completion_drop": 0.25},
        "c3": {"minimum_stop_completion_rate": 0.5, "maximum_c2_collision_rate_increase": 0.1},
        "c4": {"minimum_completion_rate": 0.5},
    }


def _relative():
    return {
        "maximum_new_collisions": 0,
        "maximum_collision_rate_increase": 0.0,
        "maximum_unsafe_episode_rate_increase": 0.125,
        "maximum_stop_violation_rate_increase": 0.125,
        "maximum_lane_failure_rate_increase": 0.125,
        "maximum_invalid_pose_rate_increase": 0.125,
        "maximum_completion_rate_drop": 0.125,
        "maximum_restart_rate_drop": 0.125,
        "maximum_mean_progress_drop_m": 0.5,
        "maximum_minimum_clearance_drop_m": 0.05,
    }


def test_curriculum_balancing_gives_equal_curriculum_mass_then_phase_mass():
    curricula, phases = [], []
    for curriculum, counts in zip(CURRICULA, ((10, 2), (5, 5), (8, 1), (7, 3), (4, 6)), strict=True):
        for phase, count in zip(("nominal", "rare"), counts, strict=True):
            curricula.extend([curriculum] * count)
            phases.extend([phase] * count)
    weights = curriculum_balanced_probabilities(curricula, phases)
    for curriculum in CURRICULA:
        mask = np.asarray(curricula) == curriculum
        assert weights[mask].sum() == pytest.approx(0.2)
        for phase in ("nominal", "rare"):
            inner = mask & (np.asarray(phases) == phase)
            assert weights[inner].sum() == pytest.approx(0.1)


def test_curriculum_balancing_refuses_missing_curriculum():
    with pytest.raises(ValueError, match="every C0-C4"):
        curriculum_balanced_probabilities(["c0", "c1"], ["nominal", "nominal"])


def test_seed_partition_is_pairwise_and_historically_disjoint():
    validate_seed_partition({"a": [180001, 180002], "b": [180101, 180102]}, [179001])
    with pytest.raises(RuntimeError, match="historical"):
        validate_seed_partition({"a": [179001]}, [179001])
    with pytest.raises(RuntimeError, match="intersects"):
        validate_seed_partition({"a": [1], "b": [1]}, [])


def test_original_failure_yields_unresolved_not_compression_fail():
    original = _summary(completion_rate=0.0)
    candidate = _summary(completion_rate=0.0)
    decision = retention_decision("c0", candidate, original, _thresholds(), _relative())
    assert decision.status == "UNRESOLVED"


def test_candidate_absolute_or_relative_regression_yields_fail():
    original = _summary()
    candidate = _summary(completion_rate=0.0)
    decision = retention_decision("c0", candidate, original, _thresholds(), _relative())
    assert decision.status == "FAIL"
    assert not decision.candidate_absolute_pass
    assert not decision.relative_pass


def test_cross_curriculum_drop_checks_use_prior_stage_summaries():
    checks = absolute_retention_checks(
        "c1", _summary(completion_rate=0.5), _thresholds(), prior_summaries={"c0": _summary(completion_rate=1.0)}
    )
    assert not checks["maximum_small_loop_completion_drop"]


def test_first_failure_event_is_earliest_and_preserves_simultaneous_labels():
    rows = [
        {"step": 0, "collision": False, "terminated": False, "truncated": False},
        {"step": 1, "collision": True, "lane_failure": True, "terminated": True, "completed": False},
        {"step": 2, "invalid_pose": True, "terminated": True, "completed": False},
    ]
    event = first_objective_failure_event(rows)
    assert event == {
        "step": 1,
        "event_labels": ["collision", "lane_failure", "termination_without_completion"],
    }


def test_historical_registry_verification_matches_real_files():
    path = ROOT / "artifacts/f12_belief_ppo_compression_v1/final/ablation_registry.json"
    entries = verify_registry(
        path,
        expected_registry_sha256="4160df2cff9162ce89288aa3a405e6f2d8ecf0578e4aa9f365db71e4acdcb91b",
        collection_key="variants",
    )
    assert tuple(entries) == tuple(f"A{index}" for index in range(8))


def test_historical_f11_f12_f13_f14_reports_are_not_f15_outputs():
    config = (ROOT / "configs/f15_cross_curriculum_recovery_v1.toml").read_text(encoding="utf-8")
    assert "artifacts/f15_cross_curriculum_recovery_v1" in config
    assert "artifacts/f14_explainability_aware_compression_v1" not in config


def test_same_state_fidelity_reports_saturation_disagreement():
    original = np.asarray([[0.2, 0.0], [0.4, 0.0], [0.1, 4.0]], dtype=np.float32)
    candidate = np.asarray([[0.2, 0.0], [0.39, 0.0], [0.1, 4.0]], dtype=np.float32)
    result = action_fidelity(original, candidate, omega_deadband=0.2)
    assert result["action_bound_saturation_frequency"]["disagreement"] == pytest.approx(1.0 / 3.0)
