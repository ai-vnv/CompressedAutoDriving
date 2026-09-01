from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path("experiments").resolve()))
import run_f11_r006_once  # noqa: E402

from duckie_pomdp.explain.confirmatory_intervention import (
    intervention_gate_results,
    paired_effect_summary,
    seed_cluster_bootstrap_ci,
)


def test_seed_cluster_bootstrap_is_deterministic_and_seed_clustered() -> None:
    values = np.asarray([1.0, 1.0, 3.0, 3.0], dtype=np.float32)
    seeds = np.asarray([10, 10, 20, 20], dtype=np.int64)
    first = seed_cluster_bootstrap_ci(
        values, seeds, replicates=1000, seed=7, confidence_level=0.95
    )
    second = seed_cluster_bootstrap_ci(
        values, seeds, replicates=1000, seed=7, confidence_level=0.95
    )
    assert first == second
    assert first[0] <= 2.0 <= first[1]


def test_paired_summary_uses_physical_delta_direction() -> None:
    result = paired_effect_summary(
        [0.1, 0.2, 0.3, 0.4],
        [1, 1, 2, 2],
        bootstrap_replicates=200,
        bootstrap_seed=3,
        confidence_level=0.95,
        direction_tolerance=1.0e-9,
    )
    assert result["mean"] == 0.25
    assert result["positive_fraction"] == 1.0
    assert result["seed_count"] == 2


def test_confirmatory_gate_keeps_sham_and_negative_controls() -> None:
    seeds = np.repeat(np.arange(8), 4)
    criteria, diagnostics = intervention_gate_results(
        pedestrian_delta_v=np.full(32, 0.2),
        pedestrian_seeds=seeds,
        pedestrian_irrelevant_delta_v=np.zeros(32),
        stop_delta_v=np.full(32, 0.1),
        stop_seeds=seeds,
        stop_control_delta_v=np.full(32, 0.001),
        lane_delta_omega=np.tile([-0.3, 0.3, -0.2, 0.2], 8),
        lane_seeds=seeds,
        lane_control_delta_omega=np.full(32, 0.02),
        sham_maximum_absolute_effect=0.0,
        gate={
            "direction_tolerance": 1.0e-8,
            "pedestrian_minimum_mean_delta_v_mps": 0.1,
            "pedestrian_nonrelevant_maximum_mean_absolute_delta_v_mps": 0.01,
            "stop_minimum_mean_delta_v_mps": 0.05,
            "stop_control_maximum_mean_absolute_delta_v_mps": 0.02,
            "lane_minimum_mean_absolute_delta_omega_rad_s": 0.1,
            "lane_minimum_curve_to_control_effect_ratio": 2.0,
            "minimum_directional_consistency": 0.75,
            "sham_action_absolute_tolerance": 1.0e-7,
        },
        bootstrap={"replicates": 200, "seed": 11, "confidence_level": 0.95},
    )
    assert all(criteria.values())
    assert diagnostics["lane_curve_to_control_effect_ratio"] == 12.5


def test_r006_preclaim_load_does_not_open_or_hash_locked_arrays(monkeypatch) -> None:
    config = Path("configs/f11_ppo_explanation_r006_v1.toml").resolve()
    original = run_f11_r006_once.sha256

    def guarded(path: Path) -> str:
        if path.name in {"locked_public_trace.npz", "final_mean_attribution.npz"}:
            raise AssertionError("locked array opened before once-only claim")
        return original(path)

    monkeypatch.setattr(run_f11_r006_once, "sha256", guarded)
    loaded = run_f11_r006_once._load(config, verify_locked_data=False)
    assert loaded[0]["data"]["once_only"] is True
