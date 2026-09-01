from __future__ import annotations

from pathlib import Path

import numpy as np

from duckie_pomdp.explain.final_attribution import (
    InsufficientReferenceSupport,
    draw_locked_same_phase_distinct_seed_references,
    mean_all_reference_attributions,
)


ROOT = Path(__file__).resolve().parents[1]


def locked_fixture(seed_count: int = 6):
    rows = []
    phases = []
    seeds = []
    for seed in range(seed_count):
        for phase_index, phase in enumerate(("nominal", "lane_curve")):
            for row in range(2):
                rows.append([seed, phase_index, row])
                phases.append(phase)
                seeds.append(100 + seed)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(phases),
        np.asarray(seeds, dtype=np.int64),
    )


def test_locked_references_use_same_phase_and_four_distinct_other_seeds() -> None:
    observations, phases, seeds = locked_fixture()
    references, indexes = draw_locked_same_phase_distinct_seed_references(
        observations,
        phases,
        seeds,
        draw_seed=2026081501,
        references_per_input=4,
        minimum_other_seed_support=4,
    )
    assert references.shape == (4, len(observations), 3)
    for row in range(len(observations)):
        assert np.all(phases[indexes[:, row]] == phases[row])
        reference_seeds = seeds[indexes[:, row]]
        assert np.all(reference_seeds != seeds[row])
        assert len(np.unique(reference_seeds)) == 4


def test_locked_reference_draw_is_deterministic() -> None:
    observations, phases, seeds = locked_fixture()
    left = draw_locked_same_phase_distinct_seed_references(
        observations, phases, seeds, draw_seed=17,
        references_per_input=4, minimum_other_seed_support=4,
    )
    right = draw_locked_same_phase_distinct_seed_references(
        observations, phases, seeds, draw_seed=17,
        references_per_input=4, minimum_other_seed_support=4,
    )
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])


def test_insufficient_phase_support_has_no_fallback() -> None:
    observations, phases, seeds = locked_fixture(seed_count=4)
    with np.testing.assert_raises(InsufficientReferenceSupport):
        draw_locked_same_phase_distinct_seed_references(
            observations,
            phases,
            seeds,
            draw_seed=17,
            references_per_input=4,
            minimum_other_seed_support=4,
        )


def test_final_estimator_is_equal_mean_of_all_six_draws() -> None:
    draws = np.arange(6 * 2 * 3 * 4, dtype=np.float32).reshape(6, 2, 3, 4)
    final = mean_all_reference_attributions(draws)
    np.testing.assert_array_equal(final, np.mean(draws, axis=0, dtype=np.float32))


def test_r004_script_is_once_only_and_has_no_privileged_read() -> None:
    source = (ROOT / "experiments" / "run_f11_r004_once.py").read_text()
    config = (ROOT / "configs" / "f11_ppo_explanation_r004_v1.toml").read_text()
    assert "once_only_launch_claim" in config
    assert "PPOCurriculumEnvironment" in source
    assert ".privileged.read(" not in source
    assert "PrivilegedSimulatorState" not in source
    assert "--mode" in source
    assert 'choices=("preflight", "once")' in source
